"""No-download smoke test for tools/analyze_weight_distribution.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from analyze_weight_distribution import synthetic_self_test


if __name__ == '__main__':
    synthetic_self_test()
