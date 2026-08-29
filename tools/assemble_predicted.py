# SPDX-License-Identifier: Apache-2.0
"""
Assemble evaluable full models from an extracted pair (master plan PoC step 4).

Builds one or more variants - fp / oracle / naive_b0 / naive_b1 / predicted (see
weight2ternary/eval_utils/assemble.py for exact definitions) - each as a loadable
HF model dir sharing the real Bonsai skeleton (embeddings/norms/head/tokenizer), so
downstream perplexity differences isolate the block weights.

Usage
-----
    python tools/assemble_predicted.py \
        --pair_dir /hdd/edwin/qwen3vsbonsai/pairs/Qwen_Qwen3-1.7B_prism-ml_Ternary-Bonsai-1.7B-unpacked \
        --bonsai_model_id prism-ml/Ternary-Bonsai-1.7B-unpacked \
        --modes fp oracle naive_b1 predicted \
        --weight_map_ckpt results/serial4/best_group_conv.pt

Outputs one dir per mode under --out_root (default /hdd/edwin/qwen3vsbonsai/assembled).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('HF_HOME', '/hdd/edwin/support/hf')
os.environ.setdefault('HF_HUB_CACHE', os.path.join(os.environ['HF_HOME'], 'hub'))

import torch


def parse_cli_args():
    from weight2ternary.eval_utils.assemble import ASSEMBLY_MODES
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pair_dir', type=str, required=True)
    parser.add_argument('--bonsai_model_id', type=str,
                        default='prism-ml/Ternary-Bonsai-1.7B-unpacked')
    parser.add_argument('--modes', type=str, nargs='+', default=['naive_b1'],
                        choices=list(ASSEMBLY_MODES))
    parser.add_argument('--weight_map_ckpt', type=str, default=None,
                        help="Trained weight-map .pt (required for mode 'predicted').")
    parser.add_argument('--out_root', type=str,
                        default='/hdd/edwin/qwen3vsbonsai/assembled')
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device for the weight-map forward pass (mode 'predicted').")
    return parser.parse_args()


def main():
    from weight2ternary.eval_utils.assemble import assemble_checkpoint
    args = parse_cli_args()
    for mode in args.modes:
        out_dir = os.path.join(args.out_root, mode)
        print(f'\n===== assembling mode={mode} -> {out_dir} =====')
        assemble_checkpoint(args.bonsai_model_id, args.pair_dir, mode, out_dir,
                            weight_map_ckpt=args.weight_map_ckpt, device=args.device,
                            group_size=args.group_size)
    return 0


if __name__ == '__main__':
    sys.exit(main())
