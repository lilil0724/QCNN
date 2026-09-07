# SPDX-License-Identifier: Apache-2.0
"""
Model tests, in order of importance:

1. The zero-init contract: an UNTRAINED model of either architecture must reproduce
   the B1 baseline code and scales EXACTLY through predict_code_and_scales - the
   master plan's decision-space-residual guarantee that training can only be judged
   by what it adds over the trivial rule.
2. Learnability smoke test: on a synthetic task whose true code is the
   group-threshold rule at tau=0.6 (so the B1/tau=0.5 baseline is wrong exactly on
   the 0.5-0.6 magnitude band), a small ContextMLP must learn to fix most of that
   band from the features within a few hundred CPU steps - the features provably
   contain the answer (|w|/group_absmean is a feature), so failure here means a
   wiring bug, not a hard research problem.

Run with:
    conda activate asr
    python tests/test_models.py
No pytest dependency - plain asserts, exits non-zero on first failure. CPU-only.
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weight2ternary.data_utils.features import (FEATURE_GROUPS, FEATURE_NAMES,
                                                FEATURE_SETS, feature_mask)
from weight2ternary.model_utils.build_model import (build_model,
                                                    group_conv_context_halo,
                                                    predict_code_and_scales)
from weight2ternary.model_utils.losses import WeightMapLoss
from synthetic_utils import make_batch

GROUP = 64  # small group so the tests stay fast; group_size is a pass-through


def test_feature_masks():
    assert len(FEATURE_NAMES) == 19
    assert int(feature_mask('full').sum()) == 19
    assert int(feature_mask('none').sum()) == 0
    assert int(feature_mask('element_only').sum()) == 4
    assert int(feature_mask('no_row').sum()) == 17
    assert set(FEATURE_SETS['no_projection']).isdisjoint(
        FEATURE_GROUPS['projection'])

    torch.manual_seed(7)
    model = build_model('context_mlp', hidden=16, group_size=GROUP,
                        feature_set='no_depth').eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.1)
    a = torch.randn(2, len(FEATURE_NAMES), 128)
    b = a.clone()
    b[:, FEATURE_NAMES.index('depth_frac')] += 100.0
    out_a = model(a)
    out_b = model(b)
    assert all(torch.allclose(x, y) for x, y in zip(out_a, out_b))


def test_group_conv_halo_matches_full_context():
    torch.manual_seed(8)
    halo = group_conv_context_halo(GROUP)
    assert halo >= 88 and halo % GROUP == 0
    model = build_model('group_conv', hidden=8, group_size=GROUP).eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.05)
    features = torch.randn(2, len(FEATURE_NAMES), 768)
    full_code, full_scale = model(features)
    start, stop = 256, 512
    segment_code, segment_scale = model(features[:, :, start - halo:stop + halo])
    assert torch.allclose(segment_code[:, :, halo:halo + stop - start],
                          full_code[:, :, start:stop], atol=1e-5, rtol=1e-5)
    g0, g1, gh = start // GROUP, stop // GROUP, halo // GROUP
    assert torch.allclose(segment_scale[:, gh:gh + g1 - g0],
                          full_scale[:, g0:g1], atol=1e-5, rtol=1e-5)

    # At the real left edge, the shorter slice preserves the model's own padding.
    edge_stop = 256
    edge_code, edge_scale = model(features[:, :, :edge_stop + halo])
    assert torch.allclose(edge_code[:, :, :edge_stop],
                          full_code[:, :, :edge_stop], atol=1e-5, rtol=1e-5)
    assert torch.allclose(edge_scale[:, :edge_stop // GROUP],
                          full_scale[:, :edge_stop // GROUP], atol=1e-5, rtol=1e-5)


def test_old_checkpoint_defaults_to_full_features():
    from weight2ternary.eval_utils.assemble import load_weight_map
    tmp_root = tempfile.mkdtemp(prefix='w2t_old_ckpt_')
    try:
        path = os.path.join(tmp_root, 'old.pt')
        model = build_model('context_mlp', hidden=8, group_size=GROUP)
        torch.save({'arch': 'context_mlp', 'hidden': 8, 'group_size': GROUP,
                    'state_dict': model.state_dict()}, path)
        loaded, loaded_group = load_weight_map(path, 'cpu')
        assert loaded.feature_set == 'full'
        assert loaded_group == GROUP
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_zero_init_equals_baseline():
    batch = make_batch(bsz=8, seg_len=256, group=GROUP, seed=0)
    for arch in ('context_mlp', 'group_conv'):
        model = build_model(arch, hidden=32, group_size=GROUP)
        model.eval()
        code, scales, logits, _ = predict_code_and_scales(model, batch)
        assert torch.equal(code, batch['baseline_code']), f'{arch}: code != baseline at init'
        assert torch.allclose(scales, batch['baseline_scales'], rtol=1e-5), \
            f'{arch}: scales != baseline at init'
        # and the margin is real: baseline logits dominate by a clear gap
        top2 = logits.topk(2, dim=1).values
        assert float((top2[:, 0] - top2[:, 1]).min().item()) > 1.0


def test_learnability_smoke():
    torch.manual_seed(0)
    model = build_model('context_mlp', hidden=32, group_size=GROUP)
    criterion = WeightMapLoss(scale_weight=0.1, hard_weight=4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batches = [make_batch(bsz=32, seg_len=256, group=GROUP, seed=s) for s in range(8)]
    for step in range(300):
        batch = batches[step % len(batches)]
        _, _, logits, log_scales = predict_code_and_scales(model, batch)
        loss, _ = criterion(logits, log_scales, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val = make_batch(bsz=64, seg_len=256, group=GROUP, seed=99)
    code, scales, _, _ = predict_code_and_scales(model, val)
    hard = val['baseline_code'] != val['code']
    hard_acc = float((code[hard] == val['code'][hard]).float().mean().item())
    overall = float((code == val['code']).float().mean().item())
    base_overall = float((val['baseline_code'] == val['code']).float().mean().item())
    print(f'  smoke: baseline acc={base_overall:.4f} model acc={overall:.4f} '
          f'hard acc={hard_acc:.4f} (hard n={int(hard.sum().item())})')
    assert overall > base_overall + 0.02, (overall, base_overall)
    assert hard_acc > 0.60, hard_acc

    # scale head learns too: true scales are 2x the baseline's ballpark by
    # construction, so trained relerr must beat the untrained baseline's
    relerr = float(((scales - val['scales']).abs() / val['scales']).mean().item())
    base_relerr = float(((val['baseline_scales'] - val['scales']).abs()
                         / val['scales']).mean().item())
    assert relerr < base_relerr, (relerr, base_relerr)


def test_loss_masking():
    batch = make_batch(bsz=8, seg_len=256, group=GROUP, seed=3)
    batch['scales'] = torch.zeros_like(batch['scales'])  # no valid scale targets
    model = build_model('context_mlp', hidden=32, group_size=GROUP)
    criterion = WeightMapLoss()
    _, _, logits, log_scales = predict_code_and_scales(model, batch)
    loss, parts = criterion(logits, log_scales, batch)
    assert parts['scale_loss'] == 0.0, 'all-zero scale targets must be fully masked'
    assert torch.isfinite(loss)


def main():
    test_feature_masks()
    test_group_conv_halo_matches_full_context()
    test_old_checkpoint_defaults_to_full_features()
    test_zero_init_equals_baseline()
    test_learnability_smoke()
    test_loss_masking()
    print('ALL test_models.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
