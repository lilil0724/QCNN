# SPDX-License-Identifier: Apache-2.0
"""
Ground-truth extraction: turn a (full-precision base, QAT low-bit) checkpoint pair
into per-layer training shards for the weight-map experiments.

For every matched 2-D transformer-block weight (same name, same shape, name contains
'.layers.') this recovers the QAT checkpoint's EXACT ternary/binary code and
per-128-column-group scales from its unpacked float weights (structure verified on
real Ternary-Bonsai-1.7B: each group's nonzero magnitudes agree to within bf16-ulp
noise - at most 0.73% relative spread on a ~1e-4 fraction of groups, bit-exact
elsewhere - so sign() gives the code and the per-group nonzero-magnitude MEDIAN the
scale), and stores them next to the base weight plus the per-row/per-column
statistics the feature builder needs, one safetensors file per layer, with a
manifest.csv indexing the lot.

Everything reads straight from safetensors shards - neither model is ever
instantiated (no AutoModel), so an 8B pair extracts comfortably on CPU.

Embeddings/lm_head are excluded by construction (the '.layers.' filter); they differ
in vocab size between Qwen3 and Bonsai anyway (151936 vs 151669) and carry none of
the ternarization structure this project studies.
"""
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .family_check import DEFAULT_GROUP_SIZE, assert_quantization_family

# projection identity (finer than compare_qat_weights.py's QKVO-vs-MLP split - the
# conditioning the master plan calls for), fixed order so one-hot encodings are stable.
PROJ_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

_LAYER_IDX_RE = re.compile(r'\.layers\.(\d+)\.')


def classify_layer_type(name: str) -> str:
    """Same buckets as tools/compare_qat_weights.py (kept name-compatible on purpose)."""
    if any(k in name for k in ('q_proj', 'k_proj', 'v_proj', 'o_proj')):
        return 'attention_qkvo'
    if any(k in name for k in ('gate_proj', 'up_proj', 'down_proj')):
        return 'mlp'
    if 'embed_tokens' in name or 'lm_head' in name:
        return 'embedding'
    return 'other'


def proj_id(name: str) -> int:
    """Index into PROJ_NAMES, or -1 for anything that is not one of the 7 projections."""
    for i, proj in enumerate(PROJ_NAMES):
        if proj in name:
            return i
    return -1


def layer_depth(name: str) -> int:
    """Decoder-layer index parsed from the parameter name, or -1 if absent."""
    m = _LAYER_IDX_RE.search(name)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Safetensors-level lazy access (no model instantiation)
# ---------------------------------------------------------------------------

def build_key_to_shard_map(snapshot_dir: str) -> dict:
    """Map parameter name -> absolute shard path for a downloaded HF snapshot,
    via model.safetensors.index.json when sharded, else the single shard's keys."""
    index_path = os.path.join(snapshot_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)['weight_map']
        return {k: os.path.join(snapshot_dir, v) for k, v in weight_map.items()}

    single = os.path.join(snapshot_dir, 'model.safetensors')
    if not os.path.exists(single):
        raise FileNotFoundError(f'No model.safetensors(.index.json) under {snapshot_dir}.')
    with safe_open(single, framework='pt') as f:
        return {k: single for k in f.keys()}


def load_tensor(key_to_shard: dict, key: str) -> torch.Tensor:
    with safe_open(key_to_shard[key], framework='pt') as f:
        return f.get_tensor(key)


def iter_matched_block_weights(base_dir: str, qat_dir: str, layer_pattern: str = None):
    """Yield (name, base_w, qat_w) for every matched 2-D '.layers.' weight pair.

    Skips (with a printed note, never silently - see CLAUDE.md's exclusion policy)
    keys missing from either side or shape-mismatched.
    """
    base_map = build_key_to_shard_map(base_dir)
    qat_map = build_key_to_shard_map(qat_dir)

    for name in sorted(base_map):
        if '.layers.' not in name or not name.endswith('.weight'):
            continue
        if layer_pattern is not None and layer_pattern not in name:
            continue
        if name not in qat_map:
            print(f'Skipping {name}: not present in QAT checkpoint.')
            continue
        base_w = load_tensor(base_map, name)
        if base_w.dim() != 2:
            continue
        qat_w = load_tensor(qat_map, name)
        if qat_w.shape != base_w.shape:
            print(f'Skipping {name}: shape mismatch base={tuple(base_w.shape)} '
                  f'qat={tuple(qat_w.shape)}.')
            continue
        yield name, base_w, qat_w


# ---------------------------------------------------------------------------
# Ground-truth code/scale recovery
# ---------------------------------------------------------------------------

def derive_code_and_scales(qat_w: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE,
                           rel_tol: float = 0.02):
    """Recover the exact {-1, 0, +1} code and per-group scales from an unpacked
    group-scaled QAT weight. Returns (code int8 [O, I], scales float32 [O, I/G],
    max_rel_dev).

    The scale is each group's MEDIAN nonzero magnitude - robust to the bf16-ulp
    noise real checkpoints carry (up to 0.73% relative spread on rare groups,
    bit-exact elsewhere; measured 2026-07-18). This is the ground-truth extractor:
    it hard-fails unless every nonzero magnitude sits within `rel_tol` of its
    group's scale, so a checkpoint that is not cleanly group-coded can never
    silently produce garbage labels. All-zero groups get scale 0 (masked out of
    scale losses downstream).
    """
    out_f, in_f = qat_w.shape
    if in_f % group_size != 0:
        raise ValueError(f'in_features={in_f} not divisible by group_size={group_size}.')

    g = qat_w.float().view(out_f, in_f // group_size, group_size)
    absg = g.abs()
    nz = g != 0
    vals = torch.where(nz, absg, torch.full_like(absg, float('nan')))
    scales = torch.nan_to_num(vals.nanmedian(dim=2).values, nan=0.0)
    code = torch.sign(g)

    dev = torch.where(nz, (absg - scales.unsqueeze(2)).abs()
                      / scales.unsqueeze(2).clamp(min=1e-12), torch.zeros_like(absg))
    max_rel_dev = float(dev.max().item())
    if max_rel_dev > rel_tol:
        raise ValueError(f'Not a clean group-size={group_size} code: max relative '
                         f'deviation from the group scale {max_rel_dev:.3e} > '
                         f'rel_tol {rel_tol:.3e}.')
    return code.view(out_f, in_f).to(torch.int8), scales, max_rel_dev


# ---------------------------------------------------------------------------
# Pair extraction to per-layer shards
# ---------------------------------------------------------------------------

def extract_pair_to_shards(base_dir: str, qat_dir: str, out_dir: str,
                           expected_family: str, group_size: int = DEFAULT_GROUP_SIZE,
                           layer_pattern: str = None):
    """Extract every matched block weight of a pair into `out_dir`.

    Per layer, one safetensors file holding: base (as stored), code (int8), scales
    (float32, 0 for all-zero groups), col_absmean/col_norm (over rows, [I]),
    row_absmean/row_norm ([O]) - the statistics features.py consumes. Plus a
    manifest.csv row per layer (name, type, proj, depth, shape, sparsity,
    tensor_absmean, recon_err, shard filename).

    The FIRST matched layer is family-asserted against `expected_family` before
    anything is written (the erratum guard); every layer is then structurally
    verified anyway by derive_code_and_scales' exactness check.
    """
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    manifest_rows = []
    family_checked = False

    for name, base_w, qat_w in iter_matched_block_weights(base_dir, qat_dir, layer_pattern):
        if not family_checked:
            assert_quantization_family(qat_w, expected_family, source=f'{qat_dir}:{name}',
                                       group_size=group_size)
            family_checked = True

        code, scales, recon_err = derive_code_and_scales(qat_w, group_size)
        base_f = base_w.float()

        shard_name = name.replace('.', '_') + '.safetensors'
        tensors = {
            'base': base_w.contiguous(),
            'code': code.contiguous(),
            'scales': scales.contiguous(),
            'col_absmean': base_f.abs().mean(dim=0).contiguous(),
            'col_norm': base_f.norm(dim=0).contiguous(),
            'row_absmean': base_f.abs().mean(dim=1).contiguous(),
            'row_norm': base_f.norm(dim=1).contiguous(),
        }
        save_file(tensors, os.path.join(out_dir, shard_name))

        manifest_rows.append({
            'layer': name,
            'family': expected_family,
            'layer_type': classify_layer_type(name),
            'proj_id': proj_id(name),
            'depth': layer_depth(name),
            'out_features': base_w.shape[0],
            'in_features': base_w.shape[1],
            'sparsity': float((code == 0).float().mean().item()),
            'tensor_absmean': float(base_f.abs().mean().item()),
            'recon_err': recon_err,
            'shard': shard_name,
        })
        print(f'{name:<55} sparsity={manifest_rows[-1]["sparsity"]:.3f} '
              f'shape={tuple(base_w.shape)}')

    if not manifest_rows:
        raise RuntimeError('No matched block weight pairs found - check the snapshot '
                           'dirs and --layer_pattern.')

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(out_dir, 'manifest.csv')
    manifest.to_csv(manifest_path, index=False)
    n_depths = manifest['depth'].nunique()
    print(f'\nExtracted {len(manifest)} layers across {n_depths} depths to {out_dir}')
    print(f'Manifest saved to {manifest_path}')
    return manifest
