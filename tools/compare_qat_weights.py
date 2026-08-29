"""
Weight-change analysis: a full-precision base model vs. its ternary-QAT counterpart
that shares the exact same architecture - e.g. Qwen/Qwen3-1.7B vs.
prism-ml/Ternary-Bonsai-1.7B-unpacked (Bonsai = Qwen3 after ternary-QAT, per the
model card).

IMPORTANT repo-naming gotcha, found the hard way (see docs/BONSAI_QWEN3_1.7B_FINDINGS.md's
erratum): `prism-ml` publishes Bonsai in two separate quantization families that
differ ONLY by an repo-name prefix - `prism-ml/Bonsai-*` (no prefix) is **binary**
({-1, +1}, no zero state - confirmed via its packed sibling repos being named
`*-mlx-1bit`), while `prism-ml/Ternary-Bonsai-*` is **ternary** ({-1, 0, +1} -
packed siblings named `*-mlx-2bit`). Using the binary repo here would silently
produce a nonsensical near-0% "ternary sparsity" result - not because of any real
ternarization-recipe difference, but simply because a binary model has no zero state
to find. Always double-check via the HF Hub API (sibling packed-variant names) rather
than trusting the base repo name alone.

`Ternary-Bonsai-*-unpacked`'s published weights ARE genuinely, cleanly ternary -
group_size=128 quantization on a row basis (confirmed directly: splitting a real
q_proj row into its 16 groups of 128 columns, every single group has exactly 1
distinct nonzero magnitude - a perfectly clean ternary code, just scaled per
128-column group rather than per whole row/tensor; the 16 distinct magnitudes seen
across a whole row is `2048 / 128`, not a sign of incomplete/continuous
quantization). The derivation below (`ternarize()`) applies a per-tensor absmean
scale rather than replicating the real group_size=128 scheme exactly, but still
produces sensible sparsity numbers here since the real zero mass dominates either
way - `compare_ternary_models.py`'s `get_ternary_code()` auto-detects the existing
zero cluster and uses `torch.sign()` directly instead, which is exactly correct
regardless of the real grouping scheme (sign is invariant to any positive per-group
scale).

Motivation (item 4b): if base and QAT weights are literally the same architecture,
comparing the DERIVED ternary code against the base weight answers "what does ternary
QAT actually DO to the weights" - does it keep large-magnitude weights and zero out
small ones (a magnitude-pruning-like behavior), does it preserve the *sign* of the
original weight where it keeps a value, and does this differ by layer type (attention
QKVO vs MLP) or structurally within a layer (some rows/columns kept denser than
others)? An explicit answer here is a step toward an analytical (rather than learned)
ternarization scheme - the same spirit as tools/analysis/analyze_hqq_scales.py
(item 4a) for HQQ scales.

Per matched Linear-weight pair (same name, same shape, in both models):
    - sparsity            : fraction of exactly-zero entries in the QAT weight.
    - sign_agreement      : of the QAT weight's NONZERO entries, what fraction share
                            the base weight's sign at that position (does ternary QAT
                            preserve direction, or does it also flip sign sometimes)?
    - kept/zeroed |base| ratio : mean(|base| at QAT-nonzero positions) /
                            mean(|base| at QAT-zero positions). >1 means QAT tends to
                            keep the base model's larger-magnitude weights nonzero -
                            i.e. behaves like magnitude pruning.
    - per-row / per-column : row (output-channel) and column (input-channel) sparsity
                            and base L2-norm arrays, plus their correlation - do
                            higher-norm (more "important") rows/columns get pruned
                            less?
Aggregated per layer-type (QKVO / MLP / other) and overall.

Usage
-----
Compare Qwen3-1.7B against its ternary-QAT Bonsai counterpart::

    python tools/compare_qat_weights.py \
        --base_model_id Qwen/Qwen3-1.7B \
        --qat_model_id prism-ml/Ternary-Bonsai-1.7B-unpacked

Synthetic self-test (no GPU/model download needed - validates the statistics only)::

    python tools/compare_qat_weights.py --synthetic

Ported from QuantizedASR's tools/analysis/compare_qat_weights.py (same logic, only the
qasr.model.bitnet_convert import was relocated - see model_loading_utils.py).
"""
import os
import re
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Layer-type classification (QKVO / MLP / other) - substring match against the
# parameter's dotted name, same style as bitnet_convert.py's skip-module matching.
# ---------------------------------------------------------------------------

_QKVO_KEYS = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
_MLP_KEYS = ['gate_proj', 'up_proj', 'down_proj']


def classify_layer_type(name: str) -> str:
    if any(k in name for k in _QKVO_KEYS):
        return 'attention_qkvo'
    if any(k in name for k in _MLP_KEYS):
        return 'mlp'
    if 'embed_tokens' in name or 'lm_head' in name:
        return 'embedding'
    return 'other'


# ---------------------------------------------------------------------------
# Core per-layer statistics (pure torch/numpy - no model dependency, self-testable)
# ---------------------------------------------------------------------------

def ternarize(w: torch.Tensor) -> torch.Tensor:
    """Derive the ACTUAL {-1, 0, +1} ternary code a QAT checkpoint's weight would use
    at inference time - NOT torch.sign(w).

    `prism-ml/Ternary-Bonsai-*-unpacked`'s published weights ARE genuinely, cleanly
    ternary - group_size=128 on a row basis (a real q_proj row's 16 distinct nonzero
    magnitudes = 2048/128, one clean magnitude per 128-column group, confirmed
    directly - not a sign of continuous/incomplete quantization). The derivation
    below applies a per-tensor absmean scale rather than replicating group_size=128
    exactly, but still gives sensible sparsity numbers here since the real zero mass
    dominates either way. The same quantization onebitllms.BitNetLinear itself uses
    at inference/training time: per-tensor absmean scaling, round, clamp to [-1, 1]
    (onebitllms/utils/quantization_utils.py's `_weight_quant`, reimplemented here to
    avoid an onebitllms import dependency for what is pure tensor math). NOTE:
    `prism-ml/Bonsai-*-unpacked` (no "Ternary-" prefix) is a DIFFERENT, binary
    checkpoint (0 exact zeros, no zero state at all) - do not confuse the two, see
    this module's top docstring.
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    return (w * scale).round().clamp(-1, 1)


def analyze_weight_pair(base: torch.Tensor, qat: torch.Tensor) -> dict:
    """Compare one matched (base, qat) weight pair. Both (out_features, in_features).

    `qat` is the QAT checkpoint's raw published weight (continuous, see `ternarize`'s
    docstring) - it is ternarized internally before comparison, since the raw
    published values are not themselves ternary.
    """
    base = base.detach().float().cpu()
    qat = qat.detach().float().cpu()

    qat_sign = ternarize(qat)            # {-1, 0, +1} - the code actually used at inference
    base_sign = torch.sign(base)
    nonzero = qat_sign != 0
    zero = ~nonzero

    n = qat.numel()
    n_nonzero = int(nonzero.sum().item())
    sparsity = 1.0 - n_nonzero / n

    sign_agreement = (float((qat_sign[nonzero] == base_sign[nonzero]).float().mean().item())
                      if n_nonzero > 0 else float('nan'))

    abs_base = base.abs()
    kept_mean_abs = float(abs_base[nonzero].mean().item()) if n_nonzero > 0 else float('nan')
    zeroed_mean_abs = float(abs_base[zero].mean().item()) if zero.any() else float('nan')
    kept_zeroed_ratio = (kept_mean_abs / zeroed_mean_abs
                        if zeroed_mean_abs and zeroed_mean_abs > 0 else float('nan'))

    # per-row (output channel) / per-column (input channel) structure
    row_sparsity = zero.float().mean(dim=1).numpy()
    row_base_norm = base.norm(dim=1).numpy()
    col_sparsity = zero.float().mean(dim=0).numpy()
    col_base_norm = base.norm(dim=0).numpy()

    def _safe_corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    return {
        'n_params': n,
        'sparsity': sparsity,
        'sign_agreement': sign_agreement,
        'kept_mean_abs_base': kept_mean_abs,
        'zeroed_mean_abs_base': zeroed_mean_abs,
        'kept_zeroed_ratio': kept_zeroed_ratio,
        'row_sparsity_vs_norm_corr': _safe_corr(row_sparsity, row_base_norm),
        'col_sparsity_vs_norm_corr': _safe_corr(col_sparsity, col_base_norm),
        'row_sparsity_std': float(row_sparsity.std()),
        'col_sparsity_std': float(col_sparsity.std()),
    }


# ---------------------------------------------------------------------------
# Model loading + matching
# ---------------------------------------------------------------------------

def load_causal_lm(model_id: str, dtype=torch.bfloat16, device_map: str = 'cpu',
                   trust_remote_code: bool = True):
    from model_loading_utils import load_config_with_remote_code_fallback, _resolve_model_class

    config, trust_remote_code = load_config_with_remote_code_fallback(
        model_id, None, trust_remote_code)
    cls = _resolve_model_class(model_id, config)
    print(f'Loading {model_id} with {cls.__name__} ...')
    return cls.from_pretrained(model_id, dtype=dtype, device_map=device_map,
                              trust_remote_code=trust_remote_code)


def matched_linear_pairs(base_model, qat_model, layer_pattern=None, regex=False):
    """Yield (name, base_weight, qat_weight) for every 2-D parameter present in both
    models under the same name and the same shape. Mismatched shapes (e.g. embed_tokens/
    lm_head when vocab sizes differ between base and QAT checkpoints) are skipped with
    a printed note rather than crashing - the transformer-block weights, where the
    actual ternarization pattern lives, are unaffected by that mismatch."""
    base_params = dict(base_model.named_parameters())
    qat_params = dict(qat_model.named_parameters())

    for name, base_w in base_params.items():
        if base_w.dim() != 2:
            continue
        if layer_pattern is not None:
            if regex and not re.search(rf'{layer_pattern}', name):
                continue
            if not regex and layer_pattern not in name:
                continue
        qat_w = qat_params.get(name)
        if qat_w is None:
            print(f'Skipping {name}: not present in QAT model.')
            continue
        if qat_w.shape != base_w.shape:
            print(f'Skipping {name}: shape mismatch base={tuple(base_w.shape)} '
                 f'qat={tuple(qat_w.shape)} (likely a vocab_size difference).')
            continue
        yield name, base_w, qat_w


# ---------------------------------------------------------------------------
# Reporting / plotting
# ---------------------------------------------------------------------------

def run_comparison(base_model, qat_model, layer_pattern, regex, results_dir):
    rows = []
    for name, base_w, qat_w in matched_linear_pairs(base_model, qat_model, layer_pattern, regex):
        stats = analyze_weight_pair(base_w, qat_w)
        stats['layer'] = name
        stats['layer_type'] = classify_layer_type(name)
        rows.append(stats)
        print(f'{name:<55} sparsity={stats["sparsity"]:.3f} '
             f'sign_agree={stats["sign_agreement"]:.3f} '
             f'kept/zeroed|base|={stats["kept_zeroed_ratio"]:.2f}')

    if not rows:
        raise RuntimeError('No matched Linear weight pairs found - check --layer_pattern '
                           'and that both models share the same architecture.')

    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'compare_qat_weights_per_layer.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nPer-layer report saved to {csv_path}')

    # per-layer-type aggregate
    agg = df.groupby('layer_type')[
        ['sparsity', 'sign_agreement', 'kept_zeroed_ratio',
         'row_sparsity_vs_norm_corr', 'col_sparsity_vs_norm_corr']
    ].mean()
    agg_path = os.path.join(results_dir, 'compare_qat_weights_by_layer_type.csv')
    agg.to_csv(agg_path)
    print(f'Per-layer-type aggregate saved to {agg_path}')
    print('\n' + '=' * 78)
    print('Per-layer-type averages')
    print('=' * 78)
    print(agg.to_string(float_format=lambda x: f'{x:.3f}'))

    _plot_summary(df, results_dir)
    return df, agg


def _plot_summary(df, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    types = sorted(df['layer_type'].unique())
    axes[0].boxplot([df.loc[df.layer_type == t, 'sparsity'] for t in types], labels=types)
    axes[0].set_title('Sparsity by layer type')
    axes[0].set_ylabel('fraction zero')

    axes[1].boxplot([df.loc[df.layer_type == t, 'sign_agreement'] for t in types], labels=types)
    axes[1].set_title('Sign agreement (kept entries) by layer type')
    axes[1].set_ylabel('fraction matching base sign')

    axes[2].hist(df['kept_zeroed_ratio'].dropna(), bins=30, color='steelblue',
                edgecolor='black', alpha=0.8)
    axes[2].axvline(1.0, color='red', linestyle='--', label='no magnitude preference')
    axes[2].set_title('mean|base| kept / mean|base| zeroed, per layer')
    axes[2].set_xlabel('ratio')
    axes[2].legend()

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'compare_qat_weights_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Summary plot saved to {out_path}')


# ---------------------------------------------------------------------------
# Synthetic self-test (validates the statistics without downloading any model)
# ---------------------------------------------------------------------------

def _synthetic_self_test():
    torch.manual_seed(0)
    out_f, in_f = 256, 512
    base = torch.randn(out_f, in_f) * 0.05

    # fabricate a "magnitude-pruning" ternary weight: keep sign of the top-50%
    # |base| entries, zero the rest - by construction this should show high
    # sign_agreement (~1.0) and kept_zeroed_ratio > 1 (kept entries are the larger ones).
    thresh = base.abs().median()
    keep_mask = base.abs() >= thresh
    qat = torch.where(keep_mask, torch.sign(base) * 0.02, torch.zeros_like(base))

    stats = analyze_weight_pair(base, qat)
    assert abs(stats['sparsity'] - 0.5) < 0.05, stats['sparsity']
    assert stats['sign_agreement'] > 0.99, stats['sign_agreement']
    assert stats['kept_zeroed_ratio'] > 1.0, stats['kept_zeroed_ratio']

    # a ternary weight built from RANDOM sign flips (not magnitude-based) should show
    # sign_agreement near 0.5 (chance) and kept/zeroed ratio near 1 (no magnitude preference)
    rand_mask = torch.rand(out_f, in_f) < 0.5
    rand_sign = torch.where(torch.rand(out_f, in_f) < 0.5, 1.0, -1.0)
    qat_random = torch.where(rand_mask, rand_sign * 0.02, torch.zeros_like(base))
    stats_random = analyze_weight_pair(base, qat_random)
    assert 0.35 < stats_random['sign_agreement'] < 0.65, stats_random['sign_agreement']
    assert 0.7 < stats_random['kept_zeroed_ratio'] < 1.4, stats_random['kept_zeroed_ratio']

    assert classify_layer_type('model.layers.0.self_attn.q_proj.weight') == 'attention_qkvo'
    assert classify_layer_type('model.layers.0.mlp.down_proj.weight') == 'mlp'
    assert classify_layer_type('model.embed_tokens.weight') == 'embedding'
    assert classify_layer_type('model.layers.0.input_layernorm.weight') == 'other'

    print('ALL compare_qat_weights.py CHECKS PASSED')
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--base_model_id', type=str, default=None,
                        help='Full-precision base model (e.g. Qwen/Qwen3-1.7B).')
    parser.add_argument('--qat_model_id', type=str, default=None,
                        help='Ternary-QAT checkpoint, same architecture (e.g. '
                             'prism-ml/Ternary-Bonsai-1.7B-unpacked - NOT '
                             'prism-ml/Bonsai-1.7B-unpacked, which is the binary '
                             'sibling, not ternary - see this module\'s docstring).')
    parser.add_argument('--layer_pattern', type=str, default=None)
    parser.add_argument('--regex_pattern', action='store_true')
    parser.add_argument('--model_dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--device_map', type=str, default='cpu')
    parser.add_argument('--results_dir', type=str, default='results')
    return parser.parse_args()


def main():
    args = parse_cli_args()
    if args.synthetic:
        return _synthetic_self_test()

    if not args.base_model_id or not args.qat_model_id:
        raise ValueError('--base_model_id and --qat_model_id are required outside --synthetic mode.')

    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
             'float32': torch.float32}[args.model_dtype]

    base_model = load_causal_lm(args.base_model_id, dtype=dtype, device_map=args.device_map)
    qat_model = load_causal_lm(args.qat_model_id, dtype=dtype, device_map=args.device_map)

    run_comparison(base_model, qat_model, args.layer_pattern, args.regex_pattern, args.results_dir)
    return 0


if __name__ == '__main__':
    main()
