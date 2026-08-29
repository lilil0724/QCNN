# SPDX-License-Identifier: Apache-2.0
"""On-the-fly functional decomposition of ternary code and group scales.

Each requested mode independently chooses the source of the ternary code and the
g128 scale: the extracted Bonsai endpoint (oracle), B1 per-group absmean
ternarization (baseline), or a trained weight-map (predicted).  All variants reuse
one Bonsai skeleton and one tokenized WikiText-2 test set.  Block weights are
overwritten in memory before each evaluation; no assembled checkpoint is saved.

Example
-------
    python tools/eval_oracle_decomposition.py \
        --pair_dir /hdd/edwin/qwen3vsbonsai/pairs/Qwen_Qwen3-1.7B_prism-ml_Ternary-Bonsai-1.7B-unpacked \
        --modes oracle_code_oracle_scale baseline_code_baseline_scale \
                predicted_code_oracle_scale predicted_code_predicted_scale \
        --weight_map_ckpt results/serial4/best_group_conv.pt \
        --results_dir results/oracle_decomposition
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hub traffic must never fall back to the remote machine's home directory.
os.environ.setdefault('HF_HOME', '/hdd/edwin/support/hf')
os.environ.setdefault('HF_HUB_CACHE', '/hdd/edwin/support/hf/hub')

import pandas as pd
import torch
from safetensors import safe_open

from weight2ternary.eval_utils.assemble import (
    DECOMPOSITION_MODES,
    DECOMPOSITION_SOURCES,
    PREDICTED_DECOMPOSITION_MODES,
    build_decomposed_block_weight,
    load_weight_map,
)


CONTRAST_SPECS = (
    ('oracle_code_gain',
     'baseline_code_baseline_scale', 'oracle_code_baseline_scale',
     'NLL(B code, B scale) - NLL(O code, B scale)'),
    ('oracle_scale_gain',
     'baseline_code_baseline_scale', 'baseline_code_oracle_scale',
     'NLL(B code, B scale) - NLL(B code, O scale)'),
    ('predicted_code_gain_with_oracle_scale',
     'baseline_code_oracle_scale', 'predicted_code_oracle_scale',
     'NLL(B code, O scale) - NLL(P code, O scale)'),
    ('predicted_scale_gap_with_oracle_code',
     'oracle_code_predicted_scale', 'oracle_code_oracle_scale',
     'NLL(O code, P scale) - NLL(O code, O scale)'),
    ('predicted_full_gain_over_baseline',
     'baseline_code_baseline_scale', 'predicted_code_predicted_scale',
     'NLL(B code, B scale) - NLL(P code, P scale)'),
    ('predicted_full_gap_to_oracle',
     'predicted_code_predicted_scale', 'oracle_code_oracle_scale',
     'NLL(P code, P scale) - NLL(O code, O scale)'),
)


def compute_nll_contrasts(per_mode_rows):
    """Return every predeclared NLL contrast whose two modes were evaluated."""
    by_mode = {row['mode']: float(row['nll']) for row in per_mode_rows}
    out = []
    for name, minuend, subtrahend, formula in CONTRAST_SPECS:
        if minuend in by_mode and subtrahend in by_mode:
            out.append({
                'contrast': name,
                'nll_difference': by_mode[minuend] - by_mode[subtrahend],
                'minuend_mode': minuend,
                'subtrahend_mode': subtrahend,
                'formula': formula,
            })
    return out


def _load_pair_layer(pair_dir, shard):
    layer = {}
    with safe_open(os.path.join(pair_dir, shard), framework='pt') as f:
        for key in f.keys():
            layer[key] = f.get_tensor(key)
    layer['base'] = layer['base'].float()
    return layer


@torch.no_grad()
def apply_decomposition_mode(model, manifest, pair_dir: str, mode: str,
                             max_depth: int, weight_map, prediction_device: str,
                             group_size: int):
    """Overwrite all manifest block weights of ``model`` for one intervention."""
    params = dict(model.named_parameters())
    replaced = 0
    for _, row in manifest.iterrows():
        name = row['layer']
        if name not in params:
            print(f'Skipping {name}: not present in skeleton.')
            continue
        layer = _load_pair_layer(pair_dir, row['shard'])
        new_w = build_decomposed_block_weight(
            mode, layer, row, max_depth, weight_map=weight_map,
            device=prediction_device, group_size=group_size)
        if new_w.shape != params[name].shape:
            raise ValueError(f'Shape mismatch for {name}: built={tuple(new_w.shape)}, '
                             f'model={tuple(params[name].shape)}.')
        params[name].copy_(new_w.to(device=params[name].device,
                                    dtype=params[name].dtype))
        replaced += 1
    if replaced != len(manifest):
        raise RuntimeError(f'Replaced only {replaced}/{len(manifest)} manifest weights.')
    return replaced


def validate_args(args):
    if len(set(args.modes)) != len(args.modes):
        raise ValueError('--modes contains duplicates; each intervention should be '
                         'evaluated once.')
    needs_prediction = any(mode in PREDICTED_DECOMPOSITION_MODES
                           for mode in args.modes)
    if needs_prediction:
        if not args.weight_map_ckpt:
            raise ValueError('The requested predicted mode(s) require '
                             '--weight_map_ckpt.')
        if not os.path.isfile(args.weight_map_ckpt):
            raise FileNotFoundError(f'Weight-map checkpoint not found: '
                                    f'{args.weight_map_ckpt}')
    if args.group_size <= 0:
        raise ValueError('--group_size must be positive.')


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tools.eval_perplexity import evaluate_token_windows, load_eval_tokens

    validate_args(args)
    manifest = pd.read_csv(os.path.join(args.pair_dir, 'manifest.csv'))
    max_depth = int(manifest['depth'].max())

    weight_map = None
    checkpoint_group = None
    if any(mode in PREDICTED_DECOMPOSITION_MODES for mode in args.modes):
        weight_map, checkpoint_group = load_weight_map(args.weight_map_ckpt,
                                                       args.device)
        if checkpoint_group != args.group_size:
            raise ValueError(f'weight-map trained with group_size={checkpoint_group}, '
                             f'evaluation requested {args.group_size}.')

    tokenizer = AutoTokenizer.from_pretrained(args.bonsai_model_id)
    windows = load_eval_tokens(tokenizer, args.seq_len, args.max_windows)
    print(f'Loaded one shared evaluation set: {windows.shape[0]} windows x '
          f'{windows.shape[1]} tokens.')

    dtype = torch.float16 if args.device.startswith('cuda') else torch.float32
    print(f'Loading shared Bonsai skeleton {args.bonsai_model_id} on {args.device} ...')
    model = AutoModelForCausalLM.from_pretrained(
        args.bonsai_model_id, dtype=dtype, device_map=args.device)
    model.eval()

    os.makedirs(args.results_dir, exist_ok=True)
    per_mode_path = os.path.join(args.results_dir,
                                 'oracle_decomposition_per_mode.csv')
    contrast_path = os.path.join(args.results_dir,
                                 'oracle_decomposition_contrasts.csv')
    args_path = os.path.join(args.results_dir, 'oracle_decomposition_args.json')
    with open(args_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    rows = []
    for mode in args.modes:
        code_source, scale_source = DECOMPOSITION_SOURCES[mode]
        print(f'\n===== mode={mode} (code={code_source}, scale={scale_source}) =====')
        replaced = apply_decomposition_mode(
            model, manifest, args.pair_dir, mode, max_depth, weight_map,
            args.device, args.group_size)
        metrics = evaluate_token_windows(model, windows, args.batch_size,
                                         args.device)
        row = {
            'mode': mode,
            'code_source': code_source,
            'scale_source': scale_source,
            **metrics,
            'replaced_weights': replaced,
            'bonsai_model_id': args.bonsai_model_id,
            'weight_map_ckpt': args.weight_map_ckpt or '',
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(per_mode_path, index=False)
        print(f'  nll={row["nll"]:.6f}  ppl={row["ppl"]:.3f}')

    contrasts = compute_nll_contrasts(rows)
    pd.DataFrame(contrasts, columns=[
        'contrast', 'nll_difference', 'minuend_mode', 'subtrahend_mode', 'formula'
    ]).to_csv(contrast_path, index=False)

    print('\n' + pd.DataFrame(rows)[['mode', 'nll', 'ppl']].to_string(index=False))
    if contrasts:
        print('\nNLL contrasts (positive means the subtrahend mode is better):')
        print(pd.DataFrame(contrasts)[['contrast', 'nll_difference']].to_string(
            index=False))
    else:
        print('\nNo complete contrast pair was selected; contrast CSV is empty.')
    print(f'\nSaved per-mode results to {per_mode_path}')
    print(f'Saved NLL contrasts to {contrast_path}')

    del model
    if weight_map is not None:
        del weight_map
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, contrasts


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pair_dir', type=str, required=True)
    parser.add_argument('--bonsai_model_id', type=str,
                        default='prism-ml/Ternary-Bonsai-1.7B-unpacked')
    parser.add_argument('--modes', type=str, nargs='+', required=True,
                        choices=list(DECOMPOSITION_MODES))
    parser.add_argument('--weight_map_ckpt', type=str, default=None,
                        help='Required only when a selected mode uses predicted '
                             'code or predicted scales.')
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_windows', type=int, default=200)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--results_dir', type=str,
                        default='results/oracle_decomposition')
    return parser.parse_args()


def main():
    run(parse_cli_args())
    return 0


if __name__ == '__main__':
    sys.exit(main())
