# SPDX-License-Identifier: Apache-2.0
"""
Sampler tests over a fabricated extracted-pair directory: layer-level split
discipline (no layer in both splits, depth rule respected), batch shapes/dtypes,
baseline consistency (the batch's baseline_code really is the B1 rule applied to the
batch's own base segment), seeded determinism, and eval-sweep coverage.

Run with:
    conda activate asr
    python tests/test_sampler.py
No pytest dependency - plain asserts, exits non-zero on first failure. CPU-only.
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weight2ternary.data_utils.features import NUM_FEATURES
from weight2ternary.data_utils.sampler import PairSegmentSampler, default_split_fn
from weight2ternary.eval_utils.baselines import ternarize_per_group_absmean
from synthetic_utils import write_synthetic_pair_dir

GROUP = 128
SEG_LEN = 256
BSZ = 8
N_DEPTHS = 8
OUT_F, IN_F = 64, 512


def main():
    tmp_root = tempfile.mkdtemp(prefix='w2t_test_sampler_')
    try:
        pair_dir = write_synthetic_pair_dir(tmp_root, n_depths=N_DEPTHS, out_f=OUT_F,
                                            in_f=IN_F, group=GROUP)

        train = PairSegmentSampler(pair_dir, 'train', SEG_LEN, BSZ, GROUP, seed=0)
        val = PairSegmentSampler(pair_dir, 'val', SEG_LEN, BSZ, GROUP, seed=0)

        # split discipline: disjoint layers, depth % 4 == 3 -> val, all layers used
        train_layers = set(train.manifest['layer'])
        val_layers = set(val.manifest['layer'])
        assert not train_layers & val_layers, 'split leakage'
        assert len(train_layers) + len(val_layers) == N_DEPTHS
        assert all(d % 4 == 3 for d in val.manifest['depth'])
        assert default_split_fn(3) == 'val' and default_split_fn(4) == 'train'

        # batch shapes/dtypes and internal consistency
        batch = train.sample_batch()
        n_groups = SEG_LEN // GROUP
        assert batch['features'].shape == (BSZ, NUM_FEATURES, SEG_LEN)
        assert batch['base'].shape == (BSZ, SEG_LEN)
        assert batch['code'].shape == (BSZ, SEG_LEN) and batch['code'].dtype == torch.long
        assert batch['scales'].shape == (BSZ, n_groups)
        assert batch['baseline_code'].dtype == torch.long
        assert torch.isfinite(batch['features']).all()
        assert set(batch['code'].unique().tolist()) <= {-1, 0, 1}
        assert torch.equal(
            ternarize_per_group_absmean(batch['base'], GROUP).long(),
            batch['baseline_code'])
        assert batch['layer'] in train_layers

        # seeded determinism: same seed -> identical first batch
        again = PairSegmentSampler(pair_dir, 'train', SEG_LEN, BSZ, GROUP, seed=0)
        b2 = again.sample_batch()
        assert batch['layer'] == b2['layer']
        assert torch.equal(batch['base'], b2['base'])
        assert torch.equal(batch['code'], b2['code'])

        # ... and a different seed changes the draw (overwhelmingly likely)
        other = PairSegmentSampler(pair_dir, 'train', SEG_LEN, BSZ, GROUP, seed=7)
        b3 = other.sample_batch()
        assert (b3['layer'] != batch['layer']) or not torch.equal(b3['base'], batch['base'])

        # eval sweep: covers every val layer, every column window, exact batch count
        n_batches = 0
        seen_layers = set()
        for batch in val.iter_eval_batches(rows_per_layer=OUT_F):
            n_batches += 1
            seen_layers.add(batch['layer'])
        assert seen_layers == val_layers
        expected = len(val_layers) * (IN_F // SEG_LEN) * (OUT_F // BSZ)
        assert n_batches == expected, (n_batches, expected)

        # misaligned segment length is a structural error
        try:
            PairSegmentSampler(pair_dir, 'train', SEG_LEN + 1, BSZ, GROUP, seed=0)
            raise AssertionError('misaligned segment_len was not rejected')
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print('ALL test_sampler.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
