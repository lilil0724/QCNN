# SPDX-License-Identifier: Apache-2.0
"""
Extract a (full-precision base, QAT low-bit) checkpoint pair into per-layer training
shards for the weight-map experiments (master plan PoC step 1).

Downloads both snapshots (safetensors only - neither model is ever instantiated),
verifies the QAT checkpoint's ACTUAL quantization family against --expected_family
(the erratum guard: prism-ml/Bonsai-* is BINARY, prism-ml/Ternary-Bonsai-* is
TERNARY - the repo name is never trusted), recovers the exact per-128-group ternary
code and scales, and writes one safetensors shard per transformer-block layer plus a
manifest.csv, under --out_dir.

Usage
-----
Extract the canonical 1.7B ternary pair::

    python tools/extract_pair.py \
        --base_model_id Qwen/Qwen3-1.7B \
        --qat_model_id prism-ml/Ternary-Bonsai-1.7B-unpacked \
        --expected_family ternary

Synthetic self-test (no downloads, validates derive/extract logic only)::

    python tools/extract_pair.py --synthetic

HF cache discipline (this box): downloads go to /hdd/edwin/support/hf - NEVER the
home directory. This script sets HF_HOME/HF_HUB_CACHE to that location itself when
they aren't already set, so a bare invocation is safe.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_HF_HOME = '/hdd/edwin/support/hf'
os.environ.setdefault('HF_HOME', DEFAULT_HF_HOME)
os.environ.setdefault('HF_HUB_CACHE', os.path.join(os.environ['HF_HOME'], 'hub'))

DEFAULT_PAIRS_ROOT = '/hdd/edwin/qwen3vsbonsai/pairs'


def default_out_dir(base_model_id: str, qat_model_id: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9.-]+', '_', f'{base_model_id}__{qat_model_id}')
    return os.path.join(DEFAULT_PAIRS_ROOT, slug)


def _synthetic_self_test():
    import torch
    from weight2ternary.data_utils.extract import (classify_layer_type, derive_code_and_scales,
                                                   layer_depth, proj_id)

    torch.manual_seed(0)
    out_f, in_f, group = 64, 512, 128
    code = torch.randint(-1, 2, (out_f, in_f)).float()
    scales = torch.rand(out_f, in_f // group) * 0.05 + 0.01
    qat = code * scales.repeat_interleave(group, dim=1)

    code_rec, scales_rec, err = derive_code_and_scales(qat, group)
    assert torch.equal(code_rec.float(), code), 'code roundtrip failed'
    kept_groups = (code != 0).view(out_f, -1, group).any(dim=2)
    assert torch.allclose(scales_rec[kept_groups], scales[kept_groups]), 'scale roundtrip failed'
    assert err == 0.0, err

    # a continuous tensor must be rejected, not silently coded
    try:
        derive_code_and_scales(torch.randn(out_f, in_f), group)
        raise AssertionError('continuous tensor was not rejected')
    except ValueError:
        pass

    assert classify_layer_type('model.layers.3.self_attn.q_proj.weight') == 'attention_qkvo'
    assert classify_layer_type('model.layers.3.mlp.down_proj.weight') == 'mlp'
    assert layer_depth('model.layers.17.mlp.up_proj.weight') == 17
    assert proj_id('model.layers.0.mlp.gate_proj.weight') == 4

    print('ALL extract_pair.py SYNTHETIC CHECKS PASSED')
    return 0


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--base_model_id', type=str, default=None,
                        help='Full-precision base model (e.g. Qwen/Qwen3-1.7B).')
    parser.add_argument('--qat_model_id', type=str, default=None,
                        help='Unpacked QAT checkpoint, same architecture (e.g. '
                             'prism-ml/Ternary-Bonsai-1.7B-unpacked).')
    parser.add_argument('--expected_family', type=str, default='ternary',
                        choices=['ternary', 'binary'],
                        help='Family the QAT checkpoint MUST structurally match.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help=f'Shard output dir (default: {DEFAULT_PAIRS_ROOT}/<pair slug>).')
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--layer_pattern', type=str, default=None,
                        help='Optional substring filter on layer names.')
    return parser.parse_args()


def main():
    args = parse_cli_args()
    if args.synthetic:
        return _synthetic_self_test()

    if not args.base_model_id or not args.qat_model_id:
        raise ValueError('--base_model_id and --qat_model_id are required outside '
                         '--synthetic mode.')

    from huggingface_hub import snapshot_download
    from weight2ternary.data_utils.extract import extract_pair_to_shards

    out_dir = args.out_dir or default_out_dir(args.base_model_id, args.qat_model_id)
    print(f'HF cache: {os.environ["HF_HUB_CACHE"]}')
    print(f'Output:   {out_dir}')

    base_dir = snapshot_download(args.base_model_id,
                                 allow_patterns=['*.safetensors*', 'config.json'])
    qat_dir = snapshot_download(args.qat_model_id,
                                allow_patterns=['*.safetensors*', 'config.json'])

    extract_pair_to_shards(base_dir, qat_dir, out_dir, args.expected_family,
                           group_size=args.group_size, layer_pattern=args.layer_pattern)
    return 0


if __name__ == '__main__':
    sys.exit(main())
