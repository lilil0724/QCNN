# SPDX-License-Identifier: Apache-2.0
"""
Baseline tests: B0 pinned numerically to tools/compare_qat_weights.py's ternarize()
rule, B1==B2(tau=0.5), the tau sweep recovering a constructed threshold, scale
estimation, and code metrics on constructed prediction/target pairs (including the
magnitude-pruning synthetic from compare_qat_weights.py's own self-test, so the two
tools' synthetic expectations stay aligned).

Run with:
    conda activate asr
    python tests/test_baselines.py
No pytest dependency - plain asserts, exits non-zero on first failure. CPU-only.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weight2ternary.eval_utils.baselines import (baseline_group_scales, binarize_sign,
                                                 code_metrics, group_absmean,
                                                 sweep_group_threshold,
                                                 ternarize_group_threshold,
                                                 ternarize_per_group_absmean,
                                                 ternarize_per_tensor_absmean)
from synthetic_utils import make_threshold_pair


def main():
    torch.manual_seed(0)
    group = 128
    base = torch.randn(128, 512) * 0.05

    # B0 == the reference rule from tools/compare_qat_weights.py (reimplemented, not
    # imported - this assert IS the coupling contract between the two)
    scale = 1.0 / base.abs().mean().clamp(min=1e-5)
    ref = (base * scale).round().clamp(-1, 1)
    assert torch.equal(ternarize_per_tensor_absmean(base), ref)

    # B1 == B2 at tau=0.5
    assert torch.equal(ternarize_per_group_absmean(base, group),
                       ternarize_group_threshold(base, 0.5, group))

    # group_absmean shape and value
    ga = group_absmean(base, group)
    assert ga.shape == (128, 512 // group)
    assert torch.allclose(ga[0, 0], base[0, :group].abs().mean())

    # tau sweep recovers a constructed threshold
    tb, tc = make_threshold_pair(out_f=64, in_f=512, group=group, tau=0.6)
    best_tau, per_tau = sweep_group_threshold([tb], [tc], group_size=group)
    assert abs(best_tau - 0.6) < 0.026, (best_tau, per_tau[best_tau])
    assert per_tau[best_tau] == 1.0, per_tau[best_tau]

    # magnitude-pruning synthetic (same construction as compare_qat_weights.py's
    # self-test): keep sign of the top-50% |base| entries -> metrics must show it
    thresh = base.abs().median()
    code = torch.where(base.abs() >= thresh, torch.sign(base), torch.zeros_like(base))
    m = code_metrics(torch.sign(base) * (base.abs() >= thresh).float(), code)
    assert m['accuracy'] == 1.0 and m['macro_f1'] == 1.0
    assert abs(m['sparsity_true'] - 0.5) < 0.05

    # hard-subset accounting: predictions correct ONLY outside the mask -> hard acc 0
    pred = code.clone()
    hard_mask = torch.zeros_like(code, dtype=torch.bool)
    hard_mask[:, :64] = True
    pred[:, :64] = -code[:, :64] + (code[:, :64] == 0).float()  # always wrong there
    m = code_metrics(pred, code, hard_mask=hard_mask)
    assert m['hard_accuracy'] == 0.0
    assert abs(m['hard_frac'] - 64 / 512) < 1e-9
    assert m['accuracy'] < 1.0

    # baseline scale estimate: kept-mean |w|, group absmean where nothing is kept
    b1 = ternarize_per_group_absmean(base, group)
    scales = baseline_group_scales(base, b1, group)
    kept0 = b1[0, :group] != 0
    assert torch.allclose(scales[0, 0], base[0, :group].abs()[kept0].mean())
    none_kept = baseline_group_scales(base, torch.zeros_like(b1), group)
    assert torch.allclose(none_kept, group_absmean(base, group))

    # binary baseline: plain sign, zero sparsity
    s = binarize_sign(base)
    assert ((s == 1) | (s == -1)).all()

    print('ALL test_baselines.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
