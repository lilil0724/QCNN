# SPDX-License-Identifier: Apache-2.0
"""
Training-framework overfit check (fast synthetic version of
tools/overfit_single_tensor.py's embed mode): a positional memorizer trained on ONE
small tensor's random ternary code, evaluated on the SAME entries, must approach
100% - the task is memorizable by construction, so anything well short of that
means the training machinery is broken and no harder result can be trusted.

Run with:
    conda activate asr
    python tests/test_overfit.py
No pytest dependency - plain asserts, exits non-zero on first failure. CPU-only.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'tools'))

from overfit_single_tensor import overfit_positional  # noqa: E402


def main():
    torch.manual_seed(0)
    n_rows, n_cols = 32, 128  # 4096 entries - tiny, CPU-fast
    base = torch.randn(n_rows, n_cols) * 0.05
    # a RANDOM code, deliberately uncorrelated with base: only memorization can win
    code = torch.randint(-1, 2, (n_rows, n_cols)).float()

    acc = overfit_positional(base, code, steps=800, lr=3e-3, device='cpu',
                             emb_dim=48, hidden=192, batch_elems=2048)
    print(f'  synthetic memorization accuracy: {acc:.4f}')
    assert acc >= 0.95, (f'memorizable task reached only {acc:.4f} - training '
                         f'framework is suspect')

    print('ALL test_overfit.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
