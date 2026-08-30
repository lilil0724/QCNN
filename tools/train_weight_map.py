# SPDX-License-Identifier: Apache-2.0
"""
Train a weight-map model (A0 ContextMLP / A1 GroupConv1D) on an extracted pair
(master plan PoC step 3, gate G2).

The model predicts a residual on top of the B1 per-group absmean baseline (zero-init
heads == the baseline exactly), trained with class-weighted cross-entropy on the
ternary code plus Huber-on-log for the group scales. Evaluation each epoch on the
held-out layers (depth % 4 == 3): OVERALL code accuracy (the gate-G2 number and the
model-selection criterion - it must beat the best tuned rule from
tools/run_baselines.py) plus accuracy on the hard subset where B1 is wrong.

Measured caution (serials 1/2, 2026-07-18): upweighting hard entries in the CE
(--hard_weight 4.0) with hard-accuracy model selection drives hard accuracy to ~0.73
while overall accuracy COLLAPSES below the baseline (0.45 vs 0.61) - the model
over-fires on easy entries. Defaults are therefore plain CE (--hard_weight 1.0) and
selection by overall accuracy; hard_weight stays available as an ablation knob.

Usage
-----
    python tools/train_weight_map.py \
        --pair_dir /home/pcs5060ti/Desktop/qcnn_data/pairs/Qwen_Qwen3-1.7B__prism-ml_Ternary-Bonsai-1.7B-unpacked \
        --arch context_mlp --serial 1

Outputs under --results_dir/serial{serial}/: train_log.csv (per-epoch), val_per_layer.csv
(final), best_{arch}.pt (state dict at best hard accuracy), args.txt.

Re-runs of the same experiment should reuse the same --serial (see CLAUDE.md: new
serials for re-runs make completed runs impossible to audit); different experiments
get different serials.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch


def batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(model, sampler, device, rows_per_layer, group_size):
    """Aggregate + per-layer val metrics: overall/hard code accuracy, macro F1 inputs,
    scale relative error - counted exactly (sums, not batch-mean-of-means)."""
    from weight2ternary.model_utils.build_model import predict_code_and_scales
    model.eval()
    per_layer = {}
    for batch in sampler.iter_eval_batches(rows_per_layer=rows_per_layer):
        batch = batch_to_device(batch, device)
        code, scales, _, _ = predict_code_and_scales(model, batch)
        true, baseline = batch['code'], batch['baseline_code']
        hard = baseline != true
        smask = batch['scales'] > 0
        s = per_layer.setdefault(batch['layer'], {
            'n': 0, 'correct': 0, 'n_hard': 0, 'hard_correct': 0,
            'base_correct': 0, 'n_scale': 0, 'scale_relerr_sum': 0.0})
        s['n'] += true.numel()
        s['correct'] += int((code == true).sum().item())
        s['base_correct'] += int((baseline == true).sum().item())
        s['n_hard'] += int(hard.sum().item())
        s['hard_correct'] += int((code[hard] == true[hard]).sum().item())
        s['n_scale'] += int(smask.sum().item())
        s['scale_relerr_sum'] += float(
            ((scales[smask] - batch['scales'][smask]).abs()
             / batch['scales'][smask]).sum().item())
    model.train()

    rows = []
    for layer, s in per_layer.items():
        rows.append({
            'layer': layer,
            'accuracy': s['correct'] / s['n'],
            'baseline_accuracy': s['base_correct'] / s['n'],
            'hard_frac': s['n_hard'] / s['n'],
            'hard_accuracy': s['hard_correct'] / max(1, s['n_hard']),
            'scale_relerr': s['scale_relerr_sum'] / max(1, s['n_scale']),
        })
    df = pd.DataFrame(rows)
    totals = {k: sum(s[k] for s in per_layer.values())
              for k in ('n', 'correct', 'base_correct', 'n_hard', 'hard_correct',
                        'n_scale', 'scale_relerr_sum')}
    agg = {
        'val_accuracy': totals['correct'] / totals['n'],
        'val_baseline_accuracy': totals['base_correct'] / totals['n'],
        'val_hard_frac': totals['n_hard'] / totals['n'],
        'val_hard_accuracy': totals['hard_correct'] / max(1, totals['n_hard']),
        'val_scale_relerr': totals['scale_relerr_sum'] / max(1, totals['n_scale']),
    }
    return agg, df


def train(args):
    from weight2ternary.data_utils.augment import (additive_noise, random_group_rescale,
                                                   random_sign_flip)
    from weight2ternary.data_utils.sampler import PairSegmentSampler, rebuild_features
    from weight2ternary.model_utils.build_model import build_model, predict_code_and_scales
    from weight2ternary.model_utils.losses import WeightMapLoss

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    torch.manual_seed(args.seed)

    run_dir = os.path.join(args.results_dir, f'serial{args.serial}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join(f'{k}={v}' for k, v in sorted(vars(args).items())))

    train_sampler = PairSegmentSampler(args.pair_dir, 'train', args.segment_len,
                                       args.batch_size, args.group_size, seed=args.seed)
    val_sampler = PairSegmentSampler(args.pair_dir, 'val', args.segment_len,
                                     args.batch_size, args.group_size, seed=args.seed)
    print(f'train layers: {len(train_sampler.manifest)}  '
          f'val layers: {len(val_sampler.manifest)}  device: {device}')

    model = build_model(args.arch, hidden=args.hidden,
                        group_size=args.group_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{args.arch}: {n_params / 1e6:.2f}M params')

    criterion = WeightMapLoss(class_weights=tuple(args.class_weights),
                              scale_weight=args.scale_weight,
                              hard_weight=args.hard_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch)
    aug_generator = torch.Generator().manual_seed(args.seed + 1)

    log_rows = []
    best_acc = -1.0
    best_path = os.path.join(run_dir, f'best_{args.arch}.pt')

    for epoch in range(args.epochs):
        t0 = time.time()
        loss_sums = {}
        for _ in range(args.steps_per_epoch):
            batch = train_sampler.sample_batch()
            if args.augment:
                batch = random_group_rescale(batch, generator=aug_generator)
                batch = random_sign_flip(batch, generator=aug_generator)
                if args.noise_sigma > 0:
                    batch = additive_noise(batch, args.noise_sigma,
                                           generator=aug_generator)
                batch = rebuild_features(batch, args.group_size)
            batch = batch_to_device(batch, device)

            _, _, code_logits, log_scales = predict_code_and_scales(model, batch)
            loss, parts = criterion(code_logits, log_scales, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            for k, v in parts.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v

        agg, val_df = evaluate(model, val_sampler, device, args.eval_rows_per_layer,
                               args.group_size)
        row = {'epoch': epoch, 'lr': scheduler.get_last_lr()[0],
               'seconds': time.time() - t0}
        row.update({k: v / args.steps_per_epoch for k, v in loss_sums.items()})
        row.update(agg)
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(os.path.join(run_dir, 'train_log.csv'), index=False)

        marker = ''
        if agg['val_accuracy'] > best_acc:
            best_acc = agg['val_accuracy']
            torch.save({'arch': args.arch, 'hidden': args.hidden,
                        'group_size': args.group_size,
                        'state_dict': model.state_dict()}, best_path)
            val_df.to_csv(os.path.join(run_dir, 'val_per_layer.csv'), index=False)
            marker = '  <- best'
        print(f'epoch {epoch:3d}  loss={row["loss"]:.4f}  '
              f'val_acc={agg["val_accuracy"]:.4f} '
              f'(baseline {agg["val_baseline_accuracy"]:.4f})  '
              f'hard_acc={agg["val_hard_accuracy"]:.4f} '
              f'(hard_frac {agg["val_hard_frac"]:.4f})  '
              f'scale_relerr={agg["val_scale_relerr"]:.4f}  '
              f'{row["seconds"]:.0f}s{marker}')

    print(f'\nBest val overall accuracy: {best_acc:.4f} (gate G2: must beat the best '
          f'tuned rule from tools/run_baselines.py). Model + per-layer report in '
          f'{run_dir}')
    return best_acc


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pair_dir', type=str, required=True)
    parser.add_argument('--arch', type=str, default='context_mlp',
                        choices=['context_mlp', 'group_conv'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--segment_len', type=int, default=512)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scale_weight', type=float, default=0.1)
    parser.add_argument('--hard_weight', type=float, default=1.0,
                        help='CE multiplier on entries where the B1 baseline is wrong '
                             '(>1 trades easy-entry accuracy for hard-entry accuracy; '
                             '4.0 measurably collapses overall accuracy - see docstring).')
    parser.add_argument('--class_weights', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='CE weights for classes (-1, 0, +1).')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--noise_sigma', type=float, default=0.0,
                        help='Relative additive-noise sigma (used with --augment).')
    parser.add_argument('--eval_rows_per_layer', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--serial', type=int, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    return parser.parse_args()


def main():
    train(parse_cli_args())
    return 0


if __name__ == '__main__':
    sys.exit(main())
