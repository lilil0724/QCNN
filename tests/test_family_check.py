# SPDX-License-Identifier: Apache-2.0
"""
Family-check tests: synthetic tensors with KNOWN quantization structure must be
classified correctly, and the erratum guard must hard-fail on a family mismatch
(the whole point - a binary checkpoint must never silently pass as ternary again).

Run with:
    conda activate asr
    python tests/test_family_check.py
No pytest dependency - plain asserts, exits non-zero on first failure. CPU-only.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weight2ternary.data_utils.family_check import (assert_quantization_family,
                                                    classify_quantization_family,
                                                    group_code_reconstruction_error)
from synthetic_utils import make_group_ternary


def main():
    torch.manual_seed(0)
    group = 128

    # clean synthetic ternary -> 'ternary', zero reconstruction error
    qat, code, _ = make_group_ternary(sparsity=0.4)
    err, zero_frac = group_code_reconstruction_error(qat, group)
    assert err == 0.0, err
    assert abs(zero_frac - float((code == 0).float().mean().item())) < 1e-9
    assert classify_quantization_family(qat, group) == 'ternary'

    # clean synthetic binary ({-1,+1} x group scales, no zero state) -> 'binary'
    gen = torch.Generator().manual_seed(1)
    sign = torch.where(torch.rand(64, 512, generator=gen) < 0.5, -1.0, 1.0)
    scales = torch.rand(64, 512 // group, generator=gen) * 0.05 + 0.01
    binary = sign * scales.repeat_interleave(group, dim=1)
    assert classify_quantization_family(binary, group) == 'binary'

    # ordinary continuous weights -> 'continuous'
    cont = torch.randn(64, 512) * 0.05
    assert classify_quantization_family(cont, group) == 'continuous'

    # the erratum guard: expecting ternary but handed the binary tensor must raise
    assert assert_quantization_family(qat, 'ternary', 'synthetic-ternary') == 'ternary'
    for w, wrong in ((binary, 'ternary'), (qat, 'binary'), (cont, 'ternary')):
        try:
            assert_quantization_family(w, wrong, 'synthetic-mismatch')
            raise AssertionError(f'family mismatch (expected={wrong}) was not caught')
        except ValueError:
            pass

    # invalid `expected` values are rejected too
    try:
        assert_quantization_family(qat, 'trinary', 'typo')
        raise AssertionError('invalid expected family was not rejected')
    except ValueError:
        pass

    # non-2D and non-divisible shapes are structural errors, not classifications
    for bad in (torch.randn(8), torch.randn(4, 100)):
        try:
            classify_quantization_family(bad, group)
            raise AssertionError(f'shape {tuple(bad.shape)} was not rejected')
        except ValueError:
            pass

    print('ALL test_family_check.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
