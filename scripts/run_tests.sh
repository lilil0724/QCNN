#!/bin/bash
# Run every weight2ternary check: the plain-assert test files plus the CLI tools'
# synthetic self-tests. No downloads, no GPU. Exits non-zero on the first failure.
#
#   conda activate asr && bash scripts/run_tests.sh
set -e
cd "$(dirname "$0")/.."

for t in tests/test_family_check.py tests/test_extract.py tests/test_baselines.py \
         tests/test_models.py tests/test_augment.py tests/test_sampler.py \
         tests/test_overfit.py tests/test_oracle_decomposition.py; do
    echo "== $t =="
    python "$t"
done

echo "== tools/extract_pair.py --synthetic =="
python tools/extract_pair.py --synthetic
echo "== tools/run_baselines.py --synthetic =="
python tools/run_baselines.py --synthetic

echo "ALL weight2ternary CHECKS PASSED"
