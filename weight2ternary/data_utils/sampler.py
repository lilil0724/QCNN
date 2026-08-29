# SPDX-License-Identifier: Apache-2.0
"""
Batch sampling over extracted pair shards (extract.py's output directory).

A sample is a group-aligned row segment: `batch_size` random rows of one layer,
`segment_len` consecutive columns starting at a group boundary. Each batch carries
the raw base segment, ground-truth code/scales, the B1 (per-group absmean) baseline
code/scales the models are residual against, the built feature tensor, and the
feature-context ingredients ('feat_ctx') needed to REBUILD features after an
augmentation transforms the base (rebuilding is mandatory - transforming the feature
tensor directly would silently desynchronize it from the base, see augment.py).

Split discipline (master plan section 2): NEVER split by random element - splits are
by whole layer. The default assigns every layer whose depth satisfies
depth % 4 == 3 to 'val' and the rest to 'train' (gate G2's held-out-layers setting);
alternatives (contiguous depth ranges, whole-scale holdout) pass through `split_fn`.

Layers load lazily into a small LRU cache - a 1.7B pair's full tensors (~4.5 GB)
usually fit, but the cap keeps 4B/8B pairs safe on the same box.
"""
import os
from collections import OrderedDict

import pandas as pd
import torch
from safetensors import safe_open

from .family_check import DEFAULT_GROUP_SIZE
from .features import build_features
from ..eval_utils.baselines import baseline_group_scales, ternarize_per_group_absmean


def default_split_fn(depth: int) -> str:
    return 'val' if depth % 4 == 3 else 'train'


def rebuild_features(batch: dict, group_size: int = DEFAULT_GROUP_SIZE) -> dict:
    """Recompute batch['features'] from batch['base'] + batch['feat_ctx'] in place
    (layer-level statistics stay those of the clean layer - segment-local features
    track the augmented base exactly, which is the intended semantics)."""
    ctx = batch['feat_ctx']
    batch['features'] = build_features(
        batch['base'], ctx['col_absmean_seg'], ctx['col_norm_seg'],
        ctx['row_absmean'], ctx['row_norm'], ctx['mean_col_norm'],
        ctx['mean_row_norm'], ctx['tensor_absmean'], ctx['depth_frac'],
        ctx['proj'], group_size=group_size)
    return batch


class PairSegmentSampler:

    def __init__(self, pair_dir: str, split: str = 'train', segment_len: int = 512,
                 batch_size: int = 64, group_size: int = DEFAULT_GROUP_SIZE,
                 seed: int = 0, split_fn=default_split_fn, max_cached_layers: int = 6):
        full_manifest = pd.read_csv(os.path.join(pair_dir, 'manifest.csv'))
        full_manifest['split'] = full_manifest['depth'].map(split_fn)
        self.manifest = full_manifest[full_manifest['split'] == split].reset_index(drop=True)
        if len(self.manifest) == 0:
            raise ValueError(f'No layers in split={split!r} under {pair_dir}.')

        self.pair_dir = pair_dir
        self.split = split
        self.segment_len = segment_len
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_depth = int(full_manifest['depth'].max())
        self.generator = torch.Generator().manual_seed(seed)
        self._cache = OrderedDict()
        self._max_cached = max_cached_layers

        if segment_len % group_size != 0:
            raise ValueError(f'segment_len={segment_len} not divisible by '
                             f'group_size={group_size}.')

    # -- layer access ------------------------------------------------------

    def _load_layer(self, row):
        shard = row['shard']
        if shard in self._cache:
            self._cache.move_to_end(shard)
            return self._cache[shard]
        tensors = {}
        with safe_open(os.path.join(self.pair_dir, shard), framework='pt') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        tensors['base'] = tensors['base'].float()
        tensors['mean_col_norm'] = float(tensors['col_norm'].mean().item())
        tensors['mean_row_norm'] = float(tensors['row_norm'].mean().item())
        self._cache[shard] = tensors
        if len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return tensors

    # -- batch construction ------------------------------------------------

    def _build_batch(self, row, layer, row_idx: torch.Tensor, col_start: int):
        seg = slice(col_start, col_start + self.segment_len)
        group_off = col_start // self.group_size
        n_groups = self.segment_len // self.group_size
        gseg = slice(group_off, group_off + n_groups)

        base_seg = layer['base'][row_idx, seg]
        # the residual baseline is family-specific: the B1 absmean rule for ternary
        # targets, plain sign for binary ones (a binary code has no zero state, so a
        # zero-producing baseline would be wrong by construction on ~40% of entries).
        # 'family' was added to the manifest later - older extractions imply ternary.
        family = row['family'] if 'family' in row else 'ternary'
        if family == 'binary':
            baseline_code = torch.sign(base_seg)
        else:
            baseline_code = ternarize_per_group_absmean(base_seg, self.group_size)
        batch = {
            'base': base_seg,                                            # [B, L]
            'code': layer['code'][row_idx, seg].long(),                  # [B, L]
            'scales': layer['scales'][row_idx, gseg].float(),            # [B, L/G]
            'baseline_code': baseline_code.long(),                       # [B, L]
            'baseline_scales': baseline_group_scales(base_seg, baseline_code,
                                                     self.group_size),   # [B, L/G]
            'layer': row['layer'],
            'feat_ctx': {
                'col_absmean_seg': layer['col_absmean'][seg],
                'col_norm_seg': layer['col_norm'][seg],
                'row_absmean': layer['row_absmean'][row_idx],
                'row_norm': layer['row_norm'][row_idx],
                'mean_col_norm': layer['mean_col_norm'],
                'mean_row_norm': layer['mean_row_norm'],
                'tensor_absmean': float(row['tensor_absmean']),
                'depth_frac': row['depth'] / max(1, self.max_depth),
                'proj': int(row['proj_id']),
            },
        }
        return rebuild_features(batch, self.group_size)

    def sample_batch(self) -> dict:
        """One random training batch: random layer, random rows, random group-aligned
        column offset."""
        i = int(torch.randint(len(self.manifest), (1,), generator=self.generator).item())
        row = self.manifest.iloc[i]
        layer = self._load_layer(row)
        out_f, in_f = layer['base'].shape

        row_idx = torch.randint(out_f, (self.batch_size,), generator=self.generator)
        n_offsets = (in_f - self.segment_len) // self.group_size + 1
        col_start = int(torch.randint(n_offsets, (1,),
                                      generator=self.generator).item()) * self.group_size
        return self._build_batch(row, layer, row_idx, col_start)

    def iter_eval_batches(self, rows_per_layer: int = 128, seed: int = 0):
        """Deterministic sweep for evaluation: for every layer in the split, a fixed
        row subsample (all rows when rows_per_layer covers them), full column range."""
        gen = torch.Generator().manual_seed(seed)
        for _, row in self.manifest.iterrows():
            layer = self._load_layer(row)
            out_f, in_f = layer['base'].shape
            if rows_per_layer >= out_f:
                row_idx = torch.arange(out_f)
            else:
                row_idx = torch.randperm(out_f, generator=gen)[:rows_per_layer]
            for col_start in range(0, in_f - self.segment_len + 1, self.segment_len):
                for b0 in range(0, len(row_idx), self.batch_size):
                    yield self._build_batch(row, layer, row_idx[b0:b0 + self.batch_size],
                                            col_start)
