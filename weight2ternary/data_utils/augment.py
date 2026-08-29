# SPDX-License-Identifier: Apache-2.0
"""
Label-consistent augmentations for weight-map training batches (master plan 6.1).

Only transformations under which the (input, target) pair remains VALID are allowed -
each augmentation transforms the base segment and co-transforms the targets exactly
as the true QAT map would (or leaves them invariant):

    - positive per-group rescale: base group *= c (c > 0) leaves the ternary code
      invariant (sign and |w|-vs-group-absmean ratios are scale-free) and multiplies
      the group's true/baseline scales by c. Enforces the scale invariance
      get_ternary_code() exploits (sign is invariant to any positive per-group scale).
    - sign flip: base segment row *= -1 flips the code target's sign, scales
      unchanged. Enforces odd symmetry of the map.
    - additive noise on the base only (targets untouched): the
      denoising-autoencoder view - the model should map a noisy neighborhood of the
      base weights to the same QAT outcome.

Row-permutation augmentation is deliberately absent for the PoC: the PoC models
(ContextMLP / GroupConv1D) are per-row by construction, so permuting sampled rows is
already a no-op; it only becomes a real (falsifying) augmentation for a future
row-adjacency model (A2 in the master plan).

Augmentations operate on the RAW batch (base/code/scales/baseline_*) and features
are rebuilt afterwards by the caller - transforming the feature tensor directly
would silently desynchronize it from the base segment.
"""
import torch

from .family_check import DEFAULT_GROUP_SIZE


def random_group_rescale(batch: dict, min_scale: float = 0.5, max_scale: float = 2.0,
                         group_size: int = DEFAULT_GROUP_SIZE,
                         generator: torch.Generator = None) -> dict:
    """Multiply every group of the base by an independent c ~ logU(min, max);
    true and baseline scales co-scale, codes untouched."""
    base = batch['base']
    bsz, seg_len = base.shape
    n_groups = seg_len // group_size

    log_c = torch.empty(bsz, n_groups).uniform_(
        float(torch.log(torch.tensor(min_scale))),
        float(torch.log(torch.tensor(max_scale))), generator=generator)
    c = log_c.exp()

    out = dict(batch)
    out['base'] = base * c.repeat_interleave(group_size, dim=1)
    out['scales'] = batch['scales'] * c
    out['baseline_scales'] = batch['baseline_scales'] * c
    return out


def random_sign_flip(batch: dict, flip_prob: float = 0.5,
                     generator: torch.Generator = None) -> dict:
    """Flip the sign of whole rows (base and code together) with prob `flip_prob`."""
    bsz = batch['base'].shape[0]
    flip = (torch.rand(bsz, 1, generator=generator) < flip_prob).float() * -2.0 + 1.0

    out = dict(batch)
    out['base'] = batch['base'] * flip
    out['code'] = (batch['code'].float() * flip).long()
    out['baseline_code'] = (batch['baseline_code'].float() * flip).long()
    return out


def additive_noise(batch: dict, sigma_rel: float = 0.02,
                   group_size: int = DEFAULT_GROUP_SIZE,
                   generator: torch.Generator = None) -> dict:
    """Add zero-mean Gaussian noise, sigma = sigma_rel * group_absmean, to the base
    only. Targets (and the baseline definitions, which stay tied to the CLEAN
    ternarization decision being learned) are untouched."""
    base = batch['base']
    bsz, seg_len = base.shape
    g_absmean = base.abs().view(bsz, -1, group_size).mean(dim=2, keepdim=True)
    sigma = (sigma_rel * g_absmean).expand(bsz, seg_len // group_size, group_size)
    noise = torch.randn(base.shape, generator=generator) * sigma.reshape(bsz, seg_len)

    out = dict(batch)
    out['base'] = base + noise
    return out
