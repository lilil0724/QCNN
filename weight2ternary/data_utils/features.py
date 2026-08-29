# SPDX-License-Identifier: Apache-2.0
"""
Per-element feature construction for the weight-map models.

Deliberately NOT image-locality features: per the master plan (section 0.4/3), the
real structure of a transformer linear weight for this problem is (element magnitude
relative to its 128-column quantization group) x (row statistics) x (column
statistics, which the findings doc showed carry the strongest sparsity correlation)
x (conditioning: projection identity, depth). Rows are permutable - nothing here
encodes row adjacency - so every feature is computable per row segment
independently, which is what lets the sampler feed arbitrary row slices.

All ratio features are log-scaled (weights span orders of magnitude across
layers/depths) and clamped to keep rare outliers from dominating; the signed
element feature keeps its sign, everything else is magnitude/statistics.
"""
import torch

from .family_check import DEFAULT_GROUP_SIZE
from .extract import PROJ_NAMES

# feature channel layout (order is a stable contract - models size themselves off it)
FEATURE_NAMES = [
    'w_over_group_absmean',        # signed, the raw decision variable of the absmean rule
    'abs_w_over_group_absmean',
    'rank_in_group',               # percentile of |w| within its group, [0, 1]
    'abs_w_over_group_absmax',
    'log_group_absmean_over_tensor',
    'log_group_absmax_over_absmean',
    'log_group_std_over_absmean',
    'log_row_absmean_over_tensor',
    'log_col_absmean_over_tensor',
    'log_col_norm_norm',           # col L2 norm over its per-layer mean
    'log_row_norm_norm',           # row L2 norm over its per-layer mean
    'depth_frac',
] + [f'proj_{p}' for p in PROJ_NAMES]

NUM_FEATURES = len(FEATURE_NAMES)

_CLAMP = 8.0
_EPS = 1e-12


def _log_ratio(a: torch.Tensor, b) -> torch.Tensor:
    return torch.log((a + _EPS) / (b + _EPS)).clamp(-_CLAMP, _CLAMP)


def build_features(base_seg: torch.Tensor, col_absmean_seg: torch.Tensor,
                   col_norm_seg: torch.Tensor, row_absmean: torch.Tensor,
                   row_norm: torch.Tensor, mean_col_norm: float, mean_row_norm: float,
                   tensor_absmean: float, depth_frac: float, proj: int,
                   group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Build the [B, NUM_FEATURES, L] feature tensor for a batch of row segments.

    base_seg        [B, L] float - group-aligned column slices of base rows.
    col_absmean_seg [L] / col_norm_seg [L] - per-column stats for the same slice.
    row_absmean     [B] / row_norm [B] - per-row stats of the sampled rows.
    mean_col_norm / mean_row_norm / tensor_absmean - per-layer scalars.
    """
    bsz, seg_len = base_seg.shape
    if seg_len % group_size != 0:
        raise ValueError(f'segment length {seg_len} not divisible by group_size={group_size}.')
    n_groups = seg_len // group_size
    device = base_seg.device  # constant features must follow the inputs' device

    w = base_seg.float()
    g = w.view(bsz, n_groups, group_size)
    abs_g = g.abs()

    g_absmean = abs_g.mean(dim=2, keepdim=True).clamp(min=_EPS)   # [B, n_groups, 1]
    g_absmax = abs_g.amax(dim=2, keepdim=True).clamp(min=_EPS)
    g_std = g.std(dim=2, keepdim=True)

    w_over_mean = (g / g_absmean).clamp(-_CLAMP, _CLAMP)
    abs_over_mean = (abs_g / g_absmean).clamp(max=_CLAMP)
    abs_over_max = abs_g / g_absmax
    # percentile of |w| within its group ([0, 1]; average of the two argsort ranks
    # would handle ties, plain argsort-of-argsort is enough at float resolution)
    rank = abs_g.argsort(dim=2).argsort(dim=2).float() / (group_size - 1)

    def _flat(x):
        return x.reshape(bsz, seg_len)

    def _group_bcast(x):
        return x.expand(bsz, n_groups, group_size).reshape(bsz, seg_len)

    feats = [
        _flat(w_over_mean),
        _flat(abs_over_mean),
        _flat(rank),
        _flat(abs_over_max),
        _group_bcast(_log_ratio(g_absmean, tensor_absmean)),
        _group_bcast(_log_ratio(g_absmax, g_absmean)),
        _group_bcast(_log_ratio(g_std, g_absmean)),
        _log_ratio(row_absmean.float(), tensor_absmean).view(bsz, 1).expand(bsz, seg_len),
        _log_ratio(col_absmean_seg.float(), tensor_absmean).view(1, seg_len).expand(bsz, seg_len),
        _log_ratio(col_norm_seg.float(), mean_col_norm).view(1, seg_len).expand(bsz, seg_len),
        _log_ratio(row_norm.float(), mean_row_norm).view(bsz, 1).expand(bsz, seg_len),
        torch.full((bsz, seg_len), float(depth_frac), device=device),
    ]
    for i in range(len(PROJ_NAMES)):
        feats.append(torch.full((bsz, seg_len), 1.0 if proj == i else 0.0,
                                device=device))

    return torch.stack(feats, dim=1)
