# SPDX-License-Identifier: Apache-2.0
"""Synthetic checks for code/scale source interventions and NLL contrasts."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weight2ternary.eval_utils.assemble as assemble
from tools.eval_oracle_decomposition import compute_nll_contrasts
from weight2ternary.eval_utils.baselines import (baseline_group_scales,
                                                 ternarize_per_group_absmean)


def test_all_decomposition_sources():
    group = 4
    base = torch.tensor([
        [-4.0, -0.1, 0.2, 3.0, -1.0, -0.2, 0.3, 2.0],
        [0.1, -2.0, 3.0, -0.2, 4.0, -0.1, 0.2, -3.0],
    ])
    oracle_code = torch.tensor([
        [-1, 0, 1, 1, 0, -1, 0, 1],
        [0, -1, 1, 0, 1, 0, 1, -1],
    ], dtype=torch.int8)
    oracle_scales = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    predicted_code = torch.tensor([
        [1, 1, 0, -1, -1, 0, 1, 0],
        [-1, 0, 0, 1, 0, 1, -1, 1],
    ], dtype=torch.float32)
    predicted_scales = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    layer = {'base': base, 'code': oracle_code, 'scales': oracle_scales}
    row = {'tensor_absmean': float(base.abs().mean()), 'depth': 0, 'proj_id': 0}

    baseline_code = ternarize_per_group_absmean(base, group).float()
    baseline_scales = baseline_group_scales(base, baseline_code, group)
    source_values = {
        'oracle': (oracle_code.float(), oracle_scales),
        'baseline': (baseline_code, baseline_scales),
        'predicted': (predicted_code, predicted_scales),
    }

    original_predict_layer = assemble.predict_layer
    assemble.predict_layer = lambda *args, **kwargs: (predicted_code, predicted_scales)
    try:
        for mode, (code_source, scale_source) in assemble.DECOMPOSITION_SOURCES.items():
            got = assemble.build_decomposed_block_weight(
                mode, layer, row, max_depth=1, weight_map=object(),
                group_size=group)
            code = source_values[code_source][0]
            scales = source_values[scale_source][1]
            expected = code * scales.repeat_interleave(group, dim=1)
            assert torch.equal(got, expected), mode
    finally:
        assemble.predict_layer = original_predict_layer


def test_predicted_mode_requires_checkpoint():
    group = 4
    base = torch.tensor([[-2.0, -0.1, 0.2, 3.0]])
    layer = {
        'base': base,
        'code': torch.tensor([[-1, 0, 0, 1]], dtype=torch.int8),
        'scales': torch.tensor([[1.5]]),
    }
    row = {'tensor_absmean': float(base.abs().mean()), 'depth': 0, 'proj_id': 0}
    try:
        assemble.build_decomposed_block_weight(
            'predicted_code_oracle_scale', layer, row, max_depth=1,
            weight_map=None, group_size=group)
        raise AssertionError('predicted mode accepted a missing weight-map')
    except ValueError as exc:
        assert 'weight-map checkpoint' in str(exc)


def test_nll_contrasts():
    rows = [
        {'mode': 'baseline_code_baseline_scale', 'nll': 10.0},
        {'mode': 'oracle_code_baseline_scale', 'nll': 7.0},
        {'mode': 'baseline_code_oracle_scale', 'nll': 8.0},
        {'mode': 'predicted_code_oracle_scale', 'nll': 6.5},
        {'mode': 'oracle_code_predicted_scale', 'nll': 5.5},
        {'mode': 'oracle_code_oracle_scale', 'nll': 5.0},
        {'mode': 'predicted_code_predicted_scale', 'nll': 6.0},
    ]
    contrasts = {row['contrast']: row['nll_difference']
                 for row in compute_nll_contrasts(rows)}
    assert contrasts == {
        'oracle_code_gain': 3.0,
        'oracle_scale_gain': 2.0,
        'predicted_code_gain_with_oracle_scale': 1.5,
        'predicted_scale_gap_with_oracle_code': 0.5,
        'predicted_full_gain_over_baseline': 4.0,
        'predicted_full_gap_to_oracle': 1.0,
    }, contrasts


def main():
    test_all_decomposition_sources()
    test_predicted_mode_requires_checkpoint()
    test_nll_contrasts()
    print('ALL test_oracle_decomposition.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
