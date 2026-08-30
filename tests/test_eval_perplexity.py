# SPDX-License-Identifier: Apache-2.0
"""Synthetic checks for perplexity evaluation data loading."""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval_perplexity import load_eval_tokens


def test_wikitext_uses_namespaced_dataset_id():
    calls = []
    fake_datasets = types.ModuleType('datasets')

    def fake_load_dataset(dataset_id, config_name, split):
        calls.append((dataset_id, config_name, split))
        return {'text': ['alpha', '', 'beta']}

    fake_datasets.load_dataset = fake_load_dataset
    previous = sys.modules.get('datasets')
    sys.modules['datasets'] = fake_datasets

    class FakeTokenizer:
        def __call__(self, text, return_tensors):
            assert text == 'alpha\n\nbeta'
            assert return_tensors == 'pt'
            return types.SimpleNamespace(input_ids=torch.arange(8).view(1, -1))

    try:
        windows = load_eval_tokens(FakeTokenizer(), seq_len=4, max_windows=2)
    finally:
        if previous is None:
            del sys.modules['datasets']
        else:
            sys.modules['datasets'] = previous

    assert calls == [('Salesforce/wikitext', 'wikitext-2-raw-v1', 'test')]
    assert windows.shape == (2, 4)


def main():
    test_wikitext_uses_namespaced_dataset_id()
    print('ALL test_eval_perplexity.py CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
