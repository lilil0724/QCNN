# SPDX-License-Identifier: Apache-2.0
"""
Non-learned ternarization baselines and code-space metrics.

The learned weight-map is only ever judged by what it adds over these:
    B0  per-tensor absmean ternarization - numerically identical to
        tools/compare_qat_weights.py's ternarize() (reimplemented here so the package
        stays standalone, matching this repo's copy-don't-couple convention; the
        equivalence is pinned by tests/test_baselines.py).
    B1  per-group (128-column, row basis) absmean ternarization - same rule at the
        QAT checkpoint's real grouping granularity.
    B2  per-group magnitude threshold with a tunable tau: keep sign(w) where
        |w| >= tau * group_absmean. B1 is exactly B2 at tau=0.5 (round-to-nearest of
        |w|/absmean crosses 0.5); tuning tau on train layers is the strongest
        rule-based baseline.

The 'hard subset' - the entries where B1 disagrees with the QAT ground truth - is
the region every learned-model result is reported on: per the findings doc the
trivial rules already explain most of the code, so overall accuracy is dominated by
easy entries and hides everything interesting.
"""
import torch

from ..data_utils.family_check import DEFAULT_GROUP_SIZE

CODE_CLASSES = (-1, 0, 1)


def ternarize_per_tensor_absmean(w: torch.Tensor) -> torch.Tensor:
    """B0: onebitllms-style per-tensor absmean scale, round, clamp to {-1, 0, +1}."""
    w = w.float()
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    return (w * scale).round().clamp(-1, 1)


def group_absmean(w: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Per-group mean |w|, groups of `group_size` columns on a row basis. [O, I/G]."""
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f'in_features={in_f} not divisible by group_size={group_size}.')
    return w.float().abs().view(out_f, in_f // group_size, group_size).mean(dim=2)


def ternarize_per_group_absmean(w: torch.Tensor,
                                group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """B1: absmean rule applied per 128-column group instead of per tensor."""
    return ternarize_group_threshold(w, tau=0.5, group_size=group_size)


def ternarize_group_threshold(w: torch.Tensor, tau: float,
                              group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """B2: code = sign(w) where |w| >= tau * group_absmean(w), else 0."""
    w = w.float()
    out_f, in_f = w.shape
    thresh = (tau * group_absmean(w, group_size)).clamp(min=1e-12)
    thresh = thresh.repeat_interleave(group_size, dim=1)
    return torch.sign(w) * (w.abs() >= thresh).float()


def binarize_sign(w: torch.Tensor) -> torch.Tensor:
    """Binary-family baseline: plain sign (ties-to-zero never fire on real floats)."""
    return torch.sign(w.float())


def baseline_group_scales(w: torch.Tensor, code: torch.Tensor,
                          group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Scale estimate per group given a code: mean |w| over the code's kept entries
    (falls back to group absmean where a group keeps nothing). [O, I/G]."""
    w = w.float()
    out_f, in_f = w.shape
    g = w.abs().view(out_f, in_f // group_size, group_size)
    kept = (code != 0).float().view(out_f, in_f // group_size, group_size)
    n_kept = kept.sum(dim=2)
    kept_mean = (g * kept).sum(dim=2) / n_kept.clamp(min=1.0)
    return torch.where(n_kept > 0, kept_mean, g.mean(dim=2))


def sweep_group_threshold(base_list, code_list, taus=None,
                          group_size: int = DEFAULT_GROUP_SIZE):
    """Tune B2's tau on (base, true-code) tensors: returns (best_tau, per-tau accs)."""
    if taus is None:
        # 0.30 .. 0.90: real Ternary-Bonsai-1.7B optimum sits ~0.8 (measured), so the
        # range must extend well past it - a boundary maximum means "widen and rerun"
        taus = [round(0.30 + 0.025 * i, 3) for i in range(25)]
    per_tau = {}
    for tau in taus:
        correct, total = 0, 0
        for base_w, code in zip(base_list, code_list):
            pred = ternarize_group_threshold(base_w, tau, group_size)
            correct += int((pred == code.float()).sum().item())
            total += code.numel()
        per_tau[tau] = correct / total
    best_tau = max(per_tau, key=per_tau.get)
    return best_tau, per_tau


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def code_metrics(pred: torch.Tensor, true: torch.Tensor,
                 hard_mask: torch.Tensor = None) -> dict:
    """Code-space metrics for one layer/batch: overall accuracy, macro F1 over the
    three classes, per-class recall, predicted/true sparsity - and accuracy on the
    `hard_mask` subset when given (canonically: where B1 disagrees with the truth)."""
    pred = pred.float().flatten()
    true = true.float().flatten()

    out = {
        'n': int(true.numel()),
        'accuracy': float((pred == true).float().mean().item()),
        'sparsity_pred': float((pred == 0).float().mean().item()),
        'sparsity_true': float((true == 0).float().mean().item()),
    }

    f1s = []
    for cls in CODE_CLASSES:
        tp = float(((pred == cls) & (true == cls)).sum().item())
        fp = float(((pred == cls) & (true != cls)).sum().item())
        fn = float(((pred != cls) & (true == cls)).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s.append(f1)
        out[f'recall_{cls}'] = recall
    out['macro_f1'] = sum(f1s) / len(f1s)

    if hard_mask is not None:
        hard_mask = hard_mask.flatten()
        n_hard = int(hard_mask.sum().item())
        out['n_hard'] = n_hard
        out['hard_frac'] = n_hard / max(1, true.numel())
        out['hard_accuracy'] = (float((pred[hard_mask] == true[hard_mask]).float().mean().item())
                                if n_hard > 0 else float('nan'))
    return out
