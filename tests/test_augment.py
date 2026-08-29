# SPDX-License-Identifier: Apache-2.0
"""
Augmentation tests: every augmentation must map a valid (input, target) pair to a
valid pair - base and targets co-transform exactly as augment.py's docstring
promises - and rebuilding features after augmentation must behave (sign flip flips
ONLY the signed feature channel; group rescale leaves all group-relative feature
channels untouched).

Run with:
    conda activate asr
    python tests/test_augment.py
No pytest dependency - plain asserts, exits non-zero on first failure. CPU-only.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weight2ternary.data_utils.augment import (additive_noise, random_group_rescale,
                                               random_sign_flip)
from weight2ternary.data_utils.features import FEATURE_NAMES
from weight2ternary.data_utils.sampler import rebuild_features
from weight2ternary.eval_utils.baselines import ternarize_per_group_absmean
from synthetic_utils import make_batch

GROUP = 64
SIGNED_FEAT = FEATURE_NAMES.index('w_over_group_absmean')
GROUP_REL_FEATS = [FEATURE_NAMES.index(n) for n in
                   ('abs_w_over_group_absmean', 'rank_in_group', 'abs_w_over_group_absmax')]


def main():
    gen = torch.Generator().manual_seed(0)
    batch = make_batch(bsz=16, seg_len=256, group=GROUP, seed=0)

    # ---- group rescale: codes invariant, scales co-scale, recon stays valid ----
    aug = random_group_rescale(batch, group_size=GROUP, generator=gen)
    assert torch.equal(aug['code'], batch['code'])
    assert torch.equal(aug['baseline_code'], batch['baseline_code'])
    c_from_scales = aug['scales'] / batch['scales']
    c_from_base = (aug['base'] / batch['base']).view(16, -1, GROUP).mean(dim=2)
    assert torch.allclose(c_from_scales, c_from_base, rtol=1e-4), \
        'scales must co-scale with the base groups'
    assert (c_from_scales > 0).all()
    # the B1 rule applied to the augmented base still gives the same code
    assert torch.equal(ternarize_per_group_absmean(aug['base'], GROUP).long(),
                       aug['baseline_code'])
    # group-relative features are invariant under the rescale
    aug = rebuild_features(aug, GROUP)
    for i in GROUP_REL_FEATS:
        assert torch.allclose(aug['features'][:, i], batch['features'][:, i],
                              atol=1e-5), FEATURE_NAMES[i]

    # ---- sign flip: base and code flip together, scales untouched ----
    batch = make_batch(bsz=16, seg_len=256, group=GROUP, seed=1)
    aug = random_sign_flip(batch, flip_prob=1.0, generator=gen)
    assert torch.equal(aug['base'], -batch['base'])
    assert torch.equal(aug['code'], -batch['code'])
    assert torch.equal(aug['baseline_code'], -batch['baseline_code'])
    assert torch.equal(aug['scales'], batch['scales'])
    aug = rebuild_features(aug, GROUP)
    assert torch.allclose(aug['features'][:, SIGNED_FEAT],
                          -batch['features'][:, SIGNED_FEAT], atol=1e-5)
    for i in GROUP_REL_FEATS:
        assert torch.allclose(aug['features'][:, i], batch['features'][:, i],
                              atol=1e-5), FEATURE_NAMES[i]

    # flip_prob=0 is the identity
    same = random_sign_flip(batch, flip_prob=0.0, generator=gen)
    assert torch.equal(same['base'], batch['base'])
    assert torch.equal(same['code'], batch['code'])

    # ---- additive noise: targets untouched, perturbation has the right scale ----
    batch = make_batch(bsz=16, seg_len=256, group=GROUP, seed=2)
    aug = additive_noise(batch, sigma_rel=0.02, group_size=GROUP, generator=gen)
    assert torch.equal(aug['code'], batch['code'])
    assert torch.equal(aug['scales'], batch['scales'])
    diff = aug['base'] - batch['base']
    g_absmean = batch['base'].abs().view(16, -1, GROUP).mean(dim=2)
    rel = diff.view(16, -1, GROUP).std(dim=2) / g_absmean
    assert 0.01 < float(rel.mean().item()) < 0.04, float(rel.mean().item())

    print('ALL test_augment.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
