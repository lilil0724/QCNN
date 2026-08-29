# SPDX-License-Identifier: Apache-2.0
"""
Quantization-family verification for Bonsai-style checkpoints.

Codifies the hard-won erratum lesson from docs/BONSAI_QWEN3_1.7B_FINDINGS.md as a
runtime assertion: `prism-ml` publishes Bonsai under two repo-name prefixes that
differ ONLY in quantization family - `prism-ml/Bonsai-*` is **binary** ({-1, +1}, no
zero state) while `prism-ml/Ternary-Bonsai-*` is **ternary** ({-1, 0, +1}). Mixing
them up silently produces nonsensical results (a binary model has no zero state for a
ternary analysis to find), so no extraction or training in this package is allowed to
proceed on a checkpoint whose ACTUAL weight structure has not been verified against
the family the caller claims it belongs to - the repo name alone is never trusted.

The check is structural, not name-based: a weight tensor belongs to a family iff
every 128-column group's nonzero magnitudes agree to within dtype-ulp-level noise
(a single group scale - the clean group-scaled code structure confirmed on real
Ternary-Bonsai weights in the findings doc), with the binary/ternary split decided
by the exact-zero fraction. Measured on the real 1.7B checkpoint (2026-07-18): most
layers are bit-exact, but a ~1e-4 fraction of groups carry up to 0.73% relative
spread (one bf16 ulp at these magnitudes - training-pipeline dtype noise, not real
structure), while a continuous tensor's within-group spread is ~100%; the 2% default
tolerance sits comfortably between the two.
"""
import torch

DEFAULT_GROUP_SIZE = 128

# exact-zero fraction thresholds separating the families. Real numbers for reference:
# Ternary-Bonsai-1.7B layers sit at ~0.37-0.41, the binary family at exactly 0.0.
BINARY_MAX_ZERO_FRAC = 0.001
TERNARY_MIN_ZERO_FRAC = 0.02


def group_code_reconstruction_error(w: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE):
    """Max RELATIVE within-group spread of nonzero magnitudes, over all
    `group_size`-column groups (row basis): 0 iff every group has exactly one
    distinct nonzero magnitude - the structure of a clean group-scaled
    binary/ternary code; ~1 for continuous weights. All-zero groups contribute 0.
    Returns (max_rel_spread, zero_fraction).
    """
    if w.dim() != 2:
        raise ValueError(f'Expected a 2-D weight tensor, got shape {tuple(w.shape)}.')
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f'in_features={in_f} not divisible by group_size={group_size}.')

    g = w.float().view(out_f, in_f // group_size, group_size)
    absg = g.abs()
    nz = g != 0
    gmax = torch.where(nz, absg, torch.zeros_like(absg)).amax(dim=2)
    gmin = torch.where(nz, absg, torch.full_like(absg, float('inf'))).amin(dim=2)
    has_nz = nz.any(dim=2)
    rel_spread = torch.zeros_like(gmax)
    rel_spread[has_nz] = ((gmax - gmin) / gmax.clamp(min=1e-12))[has_nz]
    zero_frac = float((w == 0).float().mean().item())
    return float(rel_spread.max().item()), zero_frac


def classify_quantization_family(w: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE,
                                 rel_tol: float = 0.02) -> str:
    """Classify one weight tensor as 'binary', 'ternary' or 'continuous'.

    'binary'/'ternary' require the clean group-scaled code structure (every group's
    nonzero magnitudes within `rel_tol` relative spread - the default 2% tolerates
    the up-to-0.73% bf16-ulp noise measured on the real checkpoint while a
    continuous tensor sits near 100%); they are separated by the exact-zero
    fraction. Anything else is 'continuous'.
    """
    max_spread, zero_frac = group_code_reconstruction_error(w, group_size)
    if max_spread > rel_tol:
        return 'continuous'
    if zero_frac <= BINARY_MAX_ZERO_FRAC:
        return 'binary'
    if zero_frac >= TERNARY_MIN_ZERO_FRAC:
        return 'ternary'
    # clean code structure but an ambiguous zero fraction - treat as ternary-like but
    # make the oddity visible rather than silently binning it.
    print(f'WARNING: clean group-code structure with ambiguous zero fraction '
          f'{zero_frac:.4f} (between {BINARY_MAX_ZERO_FRAC} and {TERNARY_MIN_ZERO_FRAC}).')
    return 'ternary'


def assert_quantization_family(w: torch.Tensor, expected: str, source: str = '<tensor>',
                               group_size: int = DEFAULT_GROUP_SIZE):
    """Hard-fail unless `w`'s actual structure matches the `expected` family.

    `source` is only used for the error message (pass the repo id / layer name).
    """
    if expected not in ('binary', 'ternary'):
        raise ValueError(f"expected must be 'binary' or 'ternary', got {expected!r}.")
    actual = classify_quantization_family(w, group_size)
    if actual != expected:
        raise ValueError(
            f'{source}: expected a {expected} checkpoint but the weight structure is '
            f'{actual}. Remember the two Bonsai families: prism-ml/Bonsai-* is BINARY, '
            f'prism-ml/Ternary-Bonsai-* is TERNARY (see docs/BONSAI_QWEN3_1.7B_FINDINGS.md '
            f'erratum) - and never trust the repo name without this structural check.')
    return actual
