# SPDX-License-Identifier: Apache-2.0
"""
Full-model assembly from an extracted pair + (optionally) a trained weight-map
(master plan PoC step 4 / section 4.2).

Every variant is built on the SAME skeleton - the real Bonsai checkpoint's
embeddings, norms, lm_head and tokenizer - with ONLY the transformer-block linear
weights replaced. That way perplexity differences between variants are attributable
purely to the block weights, with tokenization and the (vocab-size-differing)
embeddings held fixed:

    fp         base (Qwen3) block weights as-is - the full-precision reference.
    oracle     true code x true scales from the shards (sanity: must equal the real
               Bonsai checkpoint's own blocks bit-for-bit up to dtype).
    naive_b0   per-tensor absmean code + per-tensor kept-mean scale of the BASE
               weights - the simplest naive PTQ-to-ternary of Qwen3.
    naive_b1   per-group-128 absmean code + per-group kept-mean scales of the base
               weights - the naive PTQ baseline at the checkpoint's real granularity.
    predicted  trained weight-map's code and scales, run over every layer.

The perplexity question this feeds (the one that matters): does 'predicted' beat
'naive_b1' - i.e. does the learned map produce a functionally better ternary model
than naive mean-abs quantization - and how much of the naive->oracle gap does it
close?
"""
import os

import pandas as pd
import torch
from safetensors import safe_open

from ..data_utils.family_check import DEFAULT_GROUP_SIZE
from ..data_utils.features import build_features
from .baselines import (baseline_group_scales, ternarize_per_group_absmean,
                        ternarize_per_tensor_absmean)

ASSEMBLY_MODES = ('fp', 'oracle', 'naive_b0', 'naive_b1', 'predicted')

# Factorial code/scale interventions for functional attribution.  Keep these
# separate from ASSEMBLY_MODES: the latter are historical checkpoint-building
# names, while these modes explicitly state which source supplies each factor.
DECOMPOSITION_SOURCES = {
    'oracle_code_oracle_scale': ('oracle', 'oracle'),
    'oracle_code_baseline_scale': ('oracle', 'baseline'),
    'oracle_code_predicted_scale': ('oracle', 'predicted'),
    'baseline_code_oracle_scale': ('baseline', 'oracle'),
    'baseline_code_baseline_scale': ('baseline', 'baseline'),
    'predicted_code_oracle_scale': ('predicted', 'oracle'),
    'predicted_code_predicted_scale': ('predicted', 'predicted'),
}
DECOMPOSITION_MODES = tuple(DECOMPOSITION_SOURCES)
PREDICTED_DECOMPOSITION_MODES = tuple(
    mode for mode, sources in DECOMPOSITION_SOURCES.items()
    if 'predicted' in sources
)


def load_weight_map(ckpt_path: str, device: str):
    from ..model_utils.build_model import build_model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model = build_model(ckpt['arch'], hidden=ckpt['hidden'],
                        group_size=ckpt['group_size'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval(), ckpt['group_size']


@torch.no_grad()
def predict_layer(model, layer: dict, row, max_depth: int, device: str,
                  group_size: int = DEFAULT_GROUP_SIZE, row_chunk: int = 256):
    """Run the trained weight-map over one FULL layer (all rows, full width) in row
    chunks on `device`. Returns (code float [O, I], scales float32 [O, I/G])."""
    from ..model_utils.build_model import predict_code_and_scales

    base = layer['base'].to(device)
    out_f, in_f = base.shape
    mean_col_norm = float(layer['col_norm'].mean().item())
    mean_row_norm = float(layer['row_norm'].mean().item())
    codes, scales = [], []
    for r0 in range(0, out_f, row_chunk):
        rows = slice(r0, min(r0 + row_chunk, out_f))
        base_seg = base[rows]
        baseline_code = ternarize_per_group_absmean(base_seg, group_size)
        batch = {
            'features': build_features(
                base_seg, layer['col_absmean'].to(device), layer['col_norm'].to(device),
                layer['row_absmean'][rows].to(device), layer['row_norm'][rows].to(device),
                mean_col_norm, mean_row_norm, float(row['tensor_absmean']),
                row['depth'] / max(1, max_depth), int(row['proj_id']), group_size),
            'baseline_code': baseline_code.long(),
            'baseline_scales': baseline_group_scales(base_seg, baseline_code, group_size),
        }
        code, scale, _, _ = predict_code_and_scales(model, batch)
        codes.append(code.float().cpu())
        scales.append(scale.float().cpu())
    return torch.cat(codes), torch.cat(scales)


def build_block_weight(mode: str, layer: dict, row, max_depth: int,
                       weight_map=None, device: str = 'cpu',
                       group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """The replacement weight tensor (float32, caller casts) for one layer."""
    base = layer['base'].float()
    if mode == 'fp':
        return base
    if mode == 'oracle':
        return layer['code'].float() * layer['scales'].float().repeat_interleave(
            group_size, dim=1)
    if mode == 'naive_b0':
        code = ternarize_per_tensor_absmean(base)
        kept = code != 0
        scale = base.abs()[kept].mean() if kept.any() else base.abs().mean()
        return code * scale
    if mode == 'naive_b1':
        code = ternarize_per_group_absmean(base, group_size)
        scales = baseline_group_scales(base, code, group_size)
        return code * scales.repeat_interleave(group_size, dim=1)
    if mode == 'predicted':
        code, scales = predict_layer(weight_map, layer, row, max_depth, device,
                                     group_size)
        return code * scales.repeat_interleave(group_size, dim=1)
    raise ValueError(f'Unknown mode {mode!r} (choices: {ASSEMBLY_MODES}).')


@torch.no_grad()
def build_decomposed_block_weight(mode: str, layer: dict, row, max_depth: int,
                                  weight_map=None, device: str = 'cpu',
                                  group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Compose one block weight from independently selected code and scale sources.

    ``oracle`` is the extracted Bonsai endpoint, ``baseline`` is B1 group-wise
    absmean ternarization of the base weight, and ``predicted`` is the trained
    weight-map output.  A scale source is never recomputed after swapping code:
    holding it fixed is the intervention that makes code-vs-scale comparisons
    interpretable.
    """
    if mode not in DECOMPOSITION_SOURCES:
        raise ValueError(f'Unknown decomposition mode {mode!r} '
                         f'(choices: {DECOMPOSITION_MODES}).')

    code_source, scale_source = DECOMPOSITION_SOURCES[mode]
    base = layer['base'].float()
    sources = {
        'oracle': (layer['code'].float(), layer['scales'].float()),
    }
    if 'baseline' in (code_source, scale_source):
        baseline_code = ternarize_per_group_absmean(base, group_size)
        baseline_scales = baseline_group_scales(base, baseline_code, group_size)
        sources['baseline'] = (baseline_code.float(), baseline_scales.float())
    if 'predicted' in (code_source, scale_source):
        if weight_map is None:
            raise ValueError(f"mode={mode!r} needs a trained weight-map checkpoint.")
        predicted_code, predicted_scales = predict_layer(
            weight_map, layer, row, max_depth, device, group_size)
        sources['predicted'] = (predicted_code.float(), predicted_scales.float())

    code = sources[code_source][0]
    scales = sources[scale_source][1]
    expected_groups = code.shape[1] // group_size
    if code.shape[1] % group_size != 0 or scales.shape != (code.shape[0], expected_groups):
        raise ValueError(f'Incompatible code/scales for {mode}: code={tuple(code.shape)}, '
                         f'scales={tuple(scales.shape)}, group_size={group_size}.')
    return code * scales.repeat_interleave(group_size, dim=1)


def assemble_checkpoint(bonsai_model_id: str, pair_dir: str, mode: str, out_dir: str,
                        weight_map_ckpt: str = None, device: str = 'cpu',
                        group_size: int = DEFAULT_GROUP_SIZE):
    """Build one variant and save it as a loadable HF model dir (+ tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = pd.read_csv(os.path.join(pair_dir, 'manifest.csv'))
    max_depth = int(manifest['depth'].max())
    weight_map = None
    if mode == 'predicted':
        if not weight_map_ckpt:
            raise ValueError("mode='predicted' needs --weight_map_ckpt.")
        weight_map, ckpt_group = load_weight_map(weight_map_ckpt, device)
        if ckpt_group != group_size:
            raise ValueError(f'weight-map trained with group_size={ckpt_group}, '
                             f'assembly requested {group_size}.')

    print(f'Loading skeleton {bonsai_model_id} (cpu) ...')
    skeleton = AutoModelForCausalLM.from_pretrained(bonsai_model_id, dtype=torch.float16,
                                                    device_map='cpu')
    params = dict(skeleton.named_parameters())

    replaced = 0
    for _, row in manifest.iterrows():
        name = row['layer']
        if name not in params:
            print(f'Skipping {name}: not present in skeleton.')
            continue
        layer = {}
        with safe_open(os.path.join(pair_dir, row['shard']), framework='pt') as f:
            for key in f.keys():
                layer[key] = f.get_tensor(key)
        layer['base'] = layer['base'].float()
        new_w = build_block_weight(mode, layer, row, max_depth, weight_map, device,
                                   group_size)
        assert new_w.shape == params[name].shape, name
        params[name].data.copy_(new_w.to(params[name].dtype))
        replaced += 1
    print(f'Replaced {replaced}/{len(manifest)} block weights (mode={mode}).')

    os.makedirs(out_dir, exist_ok=True)
    skeleton.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(bonsai_model_id).save_pretrained(out_dir)
    print(f'Saved to {out_dir}')
    return out_dir
