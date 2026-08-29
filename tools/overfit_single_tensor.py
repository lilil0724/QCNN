# SPDX-License-Identifier: Apache-2.0
"""
Single-tensor overfitting check - the training-framework correctness test.

Trains a small MLP to predict the {-1, 0, +1} code of ONE real weight tensor and
evaluates on the SAME entries. Two modes, answering two different questions:

    --position_mode embed  (the framework test): inputs are learned row/column
        embeddings + the base weight value, so the task is memorizable by
        construction. If the machinery (loss, optimizer, heads) is correct, accuracy
        must approach 1.0; anything else indicates a training-framework bug, and no
        harder task can be trusted until it passes.
    --position_mode none   (the diagnostic): inputs are ONLY the package's context
        features (features.py) - position is invisible, so the score is the
        feature-information ceiling ON THE TRAIN TENSOR ITSELF. The gap between this
        and 1.0 is what position-blind features cannot express, cleanly separating
        "framework broken" from "features insufficient".

Usage (on an extracted pair)
----------------------------
    python tools/overfit_single_tensor.py \
        --pair_dir /hdd/edwin/qwen3vsbonsai/pairs/Qwen_Qwen3-1.7B_prism-ml_Ternary-Bonsai-1.7B-unpacked \
        --layer_index 0 --position_mode embed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn


class PositionalMemorizer(nn.Module):
    """Row/col embeddings + weight value -> 3-way code logits."""

    def __init__(self, n_rows: int, n_cols: int, emb_dim: int = 96, hidden: int = 512):
        super(PositionalMemorizer, self).__init__()
        self.row_emb = nn.Embedding(n_rows, emb_dim)
        self.col_emb = nn.Embedding(n_cols, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3))

    def forward(self, row_idx, col_idx, w):
        x = torch.cat([self.row_emb(row_idx), self.col_emb(col_idx),
                       w.unsqueeze(-1)], dim=-1)
        return self.mlp(x)


def overfit_positional(base, code, steps, lr, device, emb_dim, hidden, batch_elems):
    n_rows, n_cols = base.shape
    model = PositionalMemorizer(n_rows, n_cols, emb_dim, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rows = torch.arange(n_rows, device=device).view(-1, 1).expand(n_rows, n_cols).reshape(-1)
    cols = torch.arange(n_cols, device=device).view(1, -1).expand(n_rows, n_cols).reshape(-1)
    w = base.to(device).reshape(-1)
    target = (code.to(device).reshape(-1).long() + 1)

    gen = torch.Generator(device='cpu').manual_seed(0)
    for step in range(steps):
        idx = torch.randint(w.numel(), (batch_elems,), generator=gen).to(device)
        logits = model(rows[idx], cols[idx], w[idx])
        loss = nn.functional.cross_entropy(logits, target[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 10) == 0:
            print(f'  step {step:5d}  loss={float(loss.item()):.4f}')

    model.eval()
    correct = 0
    with torch.no_grad():
        for b0 in range(0, w.numel(), 1 << 20):
            sl = slice(b0, b0 + (1 << 20))
            pred = model(rows[sl], cols[sl], w[sl]).argmax(-1)
            correct += int((pred == target[sl]).sum().item())
    return correct / w.numel()


def overfit_features_only(base, code, row, layer, max_depth, steps, lr, device,
                          hidden, group_size):
    """Context-features-only overfit via the package's own ContextMLP on the SAME
    single layer: reports the feature-information ceiling on the train tensor."""
    from weight2ternary.data_utils.features import build_features
    from weight2ternary.eval_utils.baselines import (baseline_group_scales,
                                                     ternarize_per_group_absmean)
    from weight2ternary.model_utils.build_model import (build_model,
                                                        predict_code_and_scales)
    from weight2ternary.model_utils.losses import WeightMapLoss

    model = build_model('context_mlp', hidden=hidden, group_size=group_size).to(device)
    criterion = WeightMapLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mean_col_norm = float(layer['col_norm'].mean().item())
    mean_row_norm = float(layer['row_norm'].mean().item())

    def make_batch(rows_idx):
        base_seg = base[rows_idx].to(device)
        baseline_code = ternarize_per_group_absmean(base_seg, group_size)
        return {
            'features': build_features(
                base_seg, layer['col_absmean'].to(device), layer['col_norm'].to(device),
                layer['row_absmean'][rows_idx].to(device),
                layer['row_norm'][rows_idx].to(device), mean_col_norm, mean_row_norm,
                float(row['tensor_absmean']), row['depth'] / max(1, max_depth),
                int(row['proj_id']), group_size),
            'base': base_seg,
            'code': code[rows_idx].to(device).long(),
            'scales': layer['scales'][rows_idx].to(device).float(),
            'baseline_code': baseline_code.long(),
            'baseline_scales': baseline_group_scales(base_seg, baseline_code, group_size),
        }

    gen = torch.Generator().manual_seed(0)
    for step in range(steps):
        rows_idx = torch.randint(base.shape[0], (64,), generator=gen)
        batch = make_batch(rows_idx)
        _, _, logits, log_scales = predict_code_and_scales(model, batch)
        loss, _ = criterion(logits, log_scales, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 10) == 0:
            print(f'  step {step:5d}  loss={float(loss.item()):.4f}')

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b0 in range(0, base.shape[0], 256):
            batch = make_batch(torch.arange(b0, min(b0 + 256, base.shape[0])))
            pred, _, _, _ = predict_code_and_scales(model, batch)
            correct += int((pred == batch['code']).sum().item())
            total += batch['code'].numel()
    return correct / total


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pair_dir', type=str, required=True)
    parser.add_argument('--layer_index', type=int, default=0,
                        help='Row index into the manifest (which layer to overfit).')
    parser.add_argument('--position_mode', type=str, default='embed',
                        choices=['embed', 'none'])
    parser.add_argument('--max_rows', type=int, default=128,
                        help="Row subset to memorize (embed mode; keeps the task "
                             "within a small model's capacity).")
    parser.add_argument('--max_cols', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--emb_dim', type=int, default=96)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--batch_elems', type=int, default=16384)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--pass_threshold', type=float, default=0.98,
                        help='PASS bar for embed mode (none mode has no bar - it '
                             'measures a ceiling, not correctness).')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    from safetensors import safe_open
    args = parse_cli_args()

    manifest = pd.read_csv(os.path.join(args.pair_dir, 'manifest.csv'))
    row = manifest.iloc[args.layer_index]
    layer = {}
    with safe_open(os.path.join(args.pair_dir, row['shard']), framework='pt') as f:
        for key in f.keys():
            layer[key] = f.get_tensor(key)
    layer['base'] = layer['base'].float()
    print(f'Overfitting {row["layer"]} shape=({row["out_features"]}, '
          f'{row["in_features"]}) mode={args.position_mode} device={args.device}')

    if args.position_mode == 'embed':
        base = layer['base'][:args.max_rows, :args.max_cols]
        code = layer['code'][:args.max_rows, :args.max_cols].float()
        acc = overfit_positional(base, code, args.steps, args.lr, args.device,
                                 args.emb_dim, args.hidden, args.batch_elems)
        verdict = 'PASS' if acc >= args.pass_threshold else 'FAIL'
        print(f'\nembed-mode train-set accuracy on {tuple(base.shape)}: {acc:.4f} '
              f'-> {verdict} (threshold {args.pass_threshold})')
        if verdict == 'FAIL':
            print('A memorizable task did not reach the bar - suspect the training '
                  'framework (loss/optimizer/heads), NOT the research question.')
            return 1
    else:
        acc = overfit_features_only(layer['base'], layer['code'].float(), row, layer,
                                    int(manifest['depth'].max()), args.steps, args.lr,
                                    args.device, args.hidden, args.group_size)
        print(f'\nfeatures-only train-set accuracy on the full layer: {acc:.4f}')
        print('This is the feature-information CEILING on the train tensor itself - '
              'the gap to 1.0 is what position-blind context features cannot express.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
