# qwen3-ternary-bonsai-analysis

Focused analysis of `Qwen/Qwen3-1.7B` (base, full-precision) vs.
`prism-ml/Ternary-Bonsai-1.7B-unpacked` (the same architecture, ternary-QAT-trained) -
the cleanest same-architecture ternary-QAT comparison available. Spun off from the
[`QuantizedASR`](../QuantizedASR) repo, where this tool was originally built and
verified; this repo is a one-way copy (nothing here is synced back).

See `docs/BONSAI_QWEN3_1.7B_FINDINGS.md` for prior findings, including an important
checkpoint-naming gotcha - read it before running anything against a new
`prism-ml/*Bonsai*` checkpoint (there are two separate, differently-named quantization
families under that author, only one of which is actually ternary).

See `CLAUDE.md` for working conventions carried over from the source repo.

**Starting a new session/agent on this project? Read `HANDOFF.md` first** - it covers
current state, environment setup, and a list of relevant tools/methods that exist in
the source repo but weren't copied here, for whenever this project's scope grows
beyond static weight comparison.

## Usage

```bash
pip install -r requirements.txt

# Compare Qwen3-1.7B against its ternary-QAT Bonsai counterpart
python tools/compare_qat_weights.py \
    --base_model_id Qwen/Qwen3-1.7B \
    --qat_model_id prism-ml/Ternary-Bonsai-1.7B-unpacked

# Synthetic self-test (no GPU/model download needed - validates the statistics only)
python tools/compare_qat_weights.py --synthetic
```

Set `HF_HOME`/`HF_HUB_CACHE` explicitly on this box before running anything that
touches the HF Hub - do not let downloads fall through to the home directory.

## Layout

- `tools/compare_qat_weights.py` - the main comparison tool (per-layer sparsity, sign
  agreement, magnitude-pruning-like behavior; aggregated by layer type).
- `tools/model_loading_utils.py` - small model-loading helpers the tool needs
  (extracted from `QuantizedASR`'s `qasr/model/bitnet_convert.py`; no other dependency
  on that repo's `qasr` package).
- `docs/BONSAI_QWEN3_1.7B_FINDINGS.md` - prior findings for this exact model pair.
- `HANDOFF.md` - context for continuing this project in a new session, including
  relevant tools from the source repo not yet copied here.
