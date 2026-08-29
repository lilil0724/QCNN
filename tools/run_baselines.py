# SPDX-License-Identifier: Apache-2.0
"""
Gate G1: non-learned baseline numbers over an extracted pair (master plan PoC step 2).

For every extracted layer, scores the rule-based ternarization baselines against the
QAT ground-truth code:
    B0  per-tensor absmean (tools/compare_qat_weights.py's ternarize() rule)
    B1  per-group-128 absmean (the checkpoint's real granularity)
    B2  per-group magnitude threshold, tau tuned on the train split only
plus the hard-subset breakdown (entries where B1 disagrees with the truth - the only
region a learned residual model can improve). These numbers decide gate G1: how much
headroom exists above the trivial rules, and where (layer type x depth) it lives.

Usage
-----
    python tools/run_baselines.py \
        --pair_dir /hdd/edwin/qwen3vsbonsai/pairs/Qwen_Qwen3-1.7B__prism-ml_Ternary-Bonsai-1.7B-unpacked

Synthetic self-test (no extracted pair needed)::

    python tools/run_baselines.py --synthetic

Outputs (under --results_dir):
    baselines_per_layer.csv, baselines_by_layer_type.csv, printed G1 summary.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch


def _load_layer_tensors(pair_dir, shard):
    from safetensors import safe_open
    out = {}
    with safe_open(os.path.join(pair_dir, shard), framework='pt') as f:
        out['base'] = f.get_tensor('base').float()
        out['code'] = f.get_tensor('code').float()
    return out


def _subsample_rows(base, code, max_rows, seed=0):
    if base.shape[0] <= max_rows:
        return base, code
    idx = torch.randperm(base.shape[0], generator=torch.Generator().manual_seed(seed))
    return base[idx[:max_rows]], code[idx[:max_rows]]


def run(pair_dir: str, group_size: int, results_dir: str, family: str,
        sweep_rows: int, split_fn=None):
    from weight2ternary.data_utils.sampler import default_split_fn
    from weight2ternary.eval_utils.baselines import (binarize_sign, code_metrics,
                                                     sweep_group_threshold,
                                                     ternarize_group_threshold,
                                                     ternarize_per_group_absmean,
                                                     ternarize_per_tensor_absmean)
    if split_fn is None:
        split_fn = default_split_fn

    manifest = pd.read_csv(os.path.join(pair_dir, 'manifest.csv'))
    manifest['split'] = manifest['depth'].map(split_fn)

    # -- tune B2's tau on the train split only (row-subsampled for speed) ----
    best_tau = None
    if family == 'ternary':
        train_rows = manifest[manifest['split'] == 'train']
        bases, codes = [], []
        for _, row in train_rows.iterrows():
            t = _load_layer_tensors(pair_dir, row['shard'])
            b, c = _subsample_rows(t['base'], t['code'], sweep_rows)
            bases.append(b)
            codes.append(c)
        best_tau, per_tau = sweep_group_threshold(bases, codes, group_size=group_size)
        print(f'B2 tau sweep (train split, {sweep_rows} rows/layer): '
              f'best tau={best_tau} acc={per_tau[best_tau]:.4f}')
        del bases, codes

    # -- score every layer with every baseline -------------------------------
    records = []
    for _, row in manifest.iterrows():
        t = _load_layer_tensors(pair_dir, row['shard'])
        base, code = t['base'], t['code']

        if family == 'ternary':
            preds = {
                'B0_tensor_absmean': ternarize_per_tensor_absmean(base),
                'B1_group_absmean': ternarize_per_group_absmean(base, group_size),
                f'B2_group_tau{best_tau}': ternarize_group_threshold(base, best_tau,
                                                                     group_size),
            }
            hard_mask = preds['B1_group_absmean'] != code
        else:
            preds = {'B0_sign': binarize_sign(base)}
            hard_mask = preds['B0_sign'] != code

        for name, pred in preds.items():
            m = code_metrics(pred, code, hard_mask=hard_mask)
            m.update({'baseline': name, 'layer': row['layer'],
                      'layer_type': row['layer_type'], 'depth': row['depth'],
                      'split': row['split']})
            records.append(m)
        print(f'{row["layer"]:<55} ' +
              ' '.join(f'{n.split("_")[0]}={m["accuracy"]:.4f}'
                       for n, m in zip(preds, records[-len(preds):])))

    df = pd.DataFrame(records)
    os.makedirs(results_dir, exist_ok=True)
    per_layer_path = os.path.join(results_dir, 'baselines_per_layer.csv')
    df.to_csv(per_layer_path, index=False)

    agg = df.groupby(['baseline', 'split', 'layer_type'])[
        ['accuracy', 'macro_f1', 'hard_frac', 'hard_accuracy',
         'sparsity_pred', 'sparsity_true']].mean()
    agg_path = os.path.join(results_dir, 'baselines_by_layer_type.csv')
    agg.to_csv(agg_path)

    print(f'\nPer-layer report saved to {per_layer_path}')
    print(f'Aggregate saved to {agg_path}')
    print('\n' + '=' * 78)
    print('G1 SUMMARY (val split = held-out layers, depth % 4 == 3)')
    print('=' * 78)
    val = df[df['split'] == 'val'].groupby('baseline')[
        ['accuracy', 'macro_f1', 'hard_frac', 'hard_accuracy']].mean()
    print(val.to_string(float_format=lambda x: f'{x:.4f}'))
    print('\nhard_frac = fraction of entries the B1 rule already gets wrong (the '
          'learnable headroom);\nhard_accuracy of B1 is 0 by construction - other '
          'rows show how much of that\nheadroom other RULES recover; a learned model '
          'must beat the best rule here (gate G2).')
    return df


def _synthetic_self_test():
    from weight2ternary.eval_utils.baselines import (code_metrics,
                                                     ternarize_group_threshold,
                                                     ternarize_per_group_absmean,
                                                     ternarize_per_tensor_absmean)
    torch.manual_seed(0)
    out_f, in_f, group = 128, 512, 128
    base = torch.randn(out_f, in_f) * 0.05

    # a target built EXACTLY by the B1 rule must be scored perfectly by B1
    target = ternarize_per_group_absmean(base, group)
    m = code_metrics(ternarize_per_group_absmean(base, group), target)
    assert m['accuracy'] == 1.0 and m['macro_f1'] == 1.0, m

    # B1 == B2 at tau=0.5 (round-to-nearest crosses 0.5 exactly)
    assert torch.equal(ternarize_per_group_absmean(base, group),
                       ternarize_group_threshold(base, 0.5, group))

    # per-tensor absmean must reproduce the reference rule from
    # tools/compare_qat_weights.py: scale = 1/mean|w| (clamped), round, clamp
    scale = 1.0 / base.abs().mean().clamp(min=1e-5)
    ref = (base * scale).round().clamp(-1, 1)
    assert torch.equal(ternarize_per_tensor_absmean(base), ref)

    # metrics: random prediction on a random target sits near chance
    pred = torch.randint(-1, 2, (out_f, in_f)).float()
    true = torch.randint(-1, 2, (out_f, in_f)).float()
    m = code_metrics(pred, true)
    assert 0.28 < m['accuracy'] < 0.39, m['accuracy']

    print('ALL run_baselines.py SYNTHETIC CHECKS PASSED')
    return 0


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--pair_dir', type=str, default=None,
                        help='Extracted pair directory (tools/extract_pair.py output).')
    parser.add_argument('--family', type=str, default='ternary',
                        choices=['ternary', 'binary'])
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--sweep_rows', type=int, default=512,
                        help='Rows per layer used for the B2 tau sweep (speed knob; '
                             'final scoring always uses full layers).')
    parser.add_argument('--results_dir', type=str, default='results')
    return parser.parse_args()


def main():
    args = parse_cli_args()
    if args.synthetic:
        return _synthetic_self_test()
    if not args.pair_dir:
        raise ValueError('--pair_dir is required outside --synthetic mode.')
    run(args.pair_dir, args.group_size, args.results_dir, args.family, args.sweep_rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
