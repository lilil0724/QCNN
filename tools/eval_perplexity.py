# SPDX-License-Identifier: Apache-2.0
"""
Perplexity evaluation of assembled model variants (master plan section 4.2 - the
metric that matters more than any weight-space number).

Non-overlapping windows of --seq_len tokens over WikiText-2 (test split),
token-mean cross-entropy, PPL = exp(loss). All variants from
tools/assemble_predicted.py share the same tokenizer/skeleton, so their PPLs are
directly comparable and differences isolate the block weights.

Usage
-----
    python tools/eval_perplexity.py \
        --model_dirs /hdd/edwin/qwen3vsbonsai/assembled/fp \
                     /hdd/edwin/qwen3vsbonsai/assembled/oracle \
                     /hdd/edwin/qwen3vsbonsai/assembled/naive_b1 \
                     /hdd/edwin/qwen3vsbonsai/assembled/predicted \
        --results_csv results/perplexity_pair1.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('HF_HOME', '/hdd/edwin/support/hf')
os.environ.setdefault('HF_HUB_CACHE', os.path.join(os.environ['HF_HOME'], 'hub'))

import pandas as pd
import torch


def load_eval_tokens(tokenizer, seq_len: int, max_windows: int):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(t for t in ds['text'] if t.strip())
    ids = tokenizer(text, return_tensors='pt').input_ids[0]
    n_windows = min(ids.numel() // seq_len, max_windows)
    return ids[:n_windows * seq_len].view(n_windows, seq_len)


@torch.no_grad()
def evaluate_token_windows(model, windows: torch.Tensor, batch_size: int, device: str):
    """Evaluate already-tokenized windows so multiple variants see identical data."""
    total_nll, total_tokens = 0.0, 0
    for b0 in range(0, windows.shape[0], batch_size):
        batch = windows[b0:b0 + batch_size].to(device)
        out = model(input_ids=batch, labels=batch)
        n_tok = batch.numel() - batch.shape[0]  # shifted targets per row
        total_nll += float(out.loss.item()) * n_tok
        total_tokens += n_tok
    if total_tokens == 0:
        raise ValueError('No evaluation tokens: lower --seq_len or raise --max_windows.')
    loss = total_nll / total_tokens
    return {'nll': loss, 'ppl': float(torch.exp(torch.tensor(loss))),
            'n_windows': int(windows.shape[0]), 'seq_len': int(windows.shape[1])}


@torch.no_grad()
def evaluate_ppl(model_dir: str, seq_len: int, batch_size: int, device: str,
                 max_windows: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16,
                                                 device_map=device)
    model.eval()
    windows = load_eval_tokens(tokenizer, seq_len, max_windows)
    result = evaluate_token_windows(model, windows, batch_size, device)
    result['model_dir'] = model_dir
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='Assembled model dirs (or HF model ids) to score.')
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_windows', type=int, default=200)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--results_csv', type=str, default='results/perplexity.csv')
    return parser.parse_args()


def main():
    args = parse_cli_args()
    rows = []
    for model_dir in args.model_dirs:
        print(f'===== {model_dir} =====')
        row = evaluate_ppl(model_dir, args.seq_len, args.batch_size, args.device,
                           args.max_windows)
        rows.append(row)
        print(f'  ppl={row["ppl"]:.3f}  (nll={row["nll"]:.4f}, '
              f'{row["n_windows"]} windows x {args.seq_len} tokens)')

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.results_csv) or '.', exist_ok=True)
    df.to_csv(args.results_csv, index=False)
    print('\n' + df.to_string(index=False))
    print(f'\nSaved to {args.results_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
