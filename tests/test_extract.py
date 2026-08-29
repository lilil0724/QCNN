# SPDX-License-Identifier: Apache-2.0
"""
Extraction tests: exact code/scale recovery from synthetic clean group-coded
tensors, rejection of non-coded tensors, name parsing, and an end-to-end
extract_pair_to_shards run over fabricated single-shard snapshot dirs (manifest
contents, shard tensors, and per-row/col statistics all checked against
constructed truth).

Run with:
    conda activate asr
    python tests/test_extract.py
No pytest dependency - plain asserts, exits non-zero on first failure. CPU-only.
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors import safe_open

from weight2ternary.data_utils.extract import (classify_layer_type, derive_code_and_scales,
                                               extract_pair_to_shards,
                                               iter_matched_block_weights, layer_depth,
                                               proj_id)
from synthetic_utils import make_group_ternary, write_synthetic_snapshot


def test_derive_roundtrip():
    qat, code, scales = make_group_ternary(out_f=64, in_f=512, group=128)
    code_rec, scales_rec, err = derive_code_and_scales(qat, 128)
    assert err == 0.0, err
    assert torch.equal(code_rec.float(), code)
    kept = (code != 0).view(64, -1, 128).any(dim=2)
    assert torch.allclose(scales_rec[kept], scales[kept])
    assert (scales_rec[~kept] == 0).all(), 'all-zero groups must get scale 0'

    # a continuous tensor must be rejected
    try:
        derive_code_and_scales(torch.randn(64, 512), 128)
        raise AssertionError('continuous tensor was not rejected')
    except ValueError:
        pass

    # non-divisible in_features must be rejected
    try:
        derive_code_and_scales(qat[:, :500], 128)
        raise AssertionError('non-divisible in_features was not rejected')
    except ValueError:
        pass


def test_name_parsing():
    assert classify_layer_type('model.layers.0.self_attn.q_proj.weight') == 'attention_qkvo'
    assert classify_layer_type('model.layers.0.mlp.down_proj.weight') == 'mlp'
    assert classify_layer_type('model.embed_tokens.weight') == 'embedding'
    assert classify_layer_type('model.layers.0.input_layernorm.weight') == 'other'
    assert layer_depth('model.layers.17.mlp.up_proj.weight') == 17
    assert layer_depth('model.embed_tokens.weight') == -1
    assert proj_id('model.layers.2.self_attn.k_proj.weight') == 1
    assert proj_id('model.layers.2.mlp.down_proj.weight') == 6
    assert proj_id('model.embed_tokens.weight') == -1


def test_end_to_end_extraction(tmp_root):
    group = 128
    gen = torch.Generator().manual_seed(0)

    # two block layers + one embedding (must be skipped by the '.layers.' filter)
    # + one shape-mismatched block layer (must be skipped with a note, not crash)
    names = ['model.layers.0.self_attn.q_proj.weight',
             'model.layers.1.mlp.down_proj.weight']
    base_tensors, qat_tensors, truth = {}, {}, {}
    for i, name in enumerate(names):
        base = torch.randn(64, 512, generator=gen) * 0.05
        qat, code, scales = make_group_ternary(out_f=64, in_f=512, group=group, seed=i)
        base_tensors[name] = base
        qat_tensors[name] = qat
        truth[name] = (base, code)
    base_tensors['model.embed_tokens.weight'] = torch.randn(100, 32, generator=gen)
    qat_tensors['model.embed_tokens.weight'] = torch.randn(90, 32, generator=gen)
    base_tensors['model.layers.2.self_attn.v_proj.weight'] = torch.randn(64, 512, generator=gen)
    qat_tensors['model.layers.2.self_attn.v_proj.weight'] = torch.randn(32, 512, generator=gen)

    base_dir = write_synthetic_snapshot(os.path.join(tmp_root, 'base'), base_tensors)
    qat_dir = write_synthetic_snapshot(os.path.join(tmp_root, 'qat'), qat_tensors)

    matched = [n for n, _, _ in iter_matched_block_weights(base_dir, qat_dir)]
    assert matched == names, matched

    out_dir = os.path.join(tmp_root, 'pair')
    manifest = extract_pair_to_shards(base_dir, qat_dir, out_dir, 'ternary', group)
    assert len(manifest) == 2
    assert list(manifest['layer']) == names
    assert list(manifest['depth']) == [0, 1]
    assert list(manifest['layer_type']) == ['attention_qkvo', 'mlp']
    assert os.path.exists(os.path.join(out_dir, 'manifest.csv'))

    for _, row in manifest.iterrows():
        base, code = truth[row['layer']]
        with safe_open(os.path.join(out_dir, row['shard']), framework='pt') as f:
            assert torch.equal(f.get_tensor('base'), base)
            assert torch.equal(f.get_tensor('code').float(), code)
            assert torch.allclose(f.get_tensor('col_absmean'), base.abs().mean(dim=0))
            assert torch.allclose(f.get_tensor('row_norm'), base.norm(dim=1))
        assert abs(row['sparsity'] - float((code == 0).float().mean().item())) < 1e-9
        assert abs(row['tensor_absmean'] - float(base.abs().mean().item())) < 1e-6

    # family guard: extracting the same pair as 'binary' must fail before writing
    try:
        extract_pair_to_shards(base_dir, qat_dir, os.path.join(tmp_root, 'pair2'),
                               'binary', group)
        raise AssertionError('ternary checkpoint accepted as binary')
    except ValueError:
        pass


def main():
    test_derive_roundtrip()
    test_name_parsing()
    tmp_root = tempfile.mkdtemp(prefix='w2t_test_extract_')
    try:
        test_end_to_end_extraction(tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print('ALL test_extract.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
