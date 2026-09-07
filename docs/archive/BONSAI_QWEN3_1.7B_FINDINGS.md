# BONSAI_QWEN3_1.7B_FINDINGS.md

Findings specific to `prism-ml/Ternary-Bonsai-1.7B-unpacked` vs. its base model
`Qwen/Qwen3-1.7B` - the smallest pair in the Bonsai family, and the cleanest
same-architecture ternary-QAT comparison available (see `CLAUDE_CHANGES.md` for the
full session history; this document collects and interprets those findings in one
place, specifically for this model pair, rather than spread across per-tool changelog
entries).

Tools used: `tools/analysis/compare_qat_weights.py` (Qwen3 vs. Bonsai, matched-layer
diff) and `tools/analysis/compare_ternary_models.py` (Bonsai vs. three other
ternary/BitNet families, architecture-agnostic stats). All numbers below are from real
GPU runs against the actual HF Hub checkpoints, not estimates.

---

## ERRATUM (read this first)

An earlier version of this document analyzed **`prism-ml/Bonsai-1.7B-unpacked`**
(no "Ternary-" prefix) and reported a large, "unexplained" sparsity anomaly relative
to Falcon-E/MS-BitNet/HF-BitNet, plus a follow-up study quantifying a spurious
~2x scale-factor gap. **That entire analysis used the wrong checkpoint.**

`prism-ml` publishes Bonsai in two separate quantization families, distinguished only
by the repo name prefix - confirmed via the HF Hub API (`list_models(author='prism-ml')`)
by checking the packed/quantized sibling variants of each:
- `prism-ml/Bonsai-*` (no prefix) - siblings are named `*-mlx-1bit` -> **binary**
  ({-1, +1}, no zero state at all).
- `prism-ml/Ternary-Bonsai-*` - siblings are named `*-mlx-2bit` -> **ternary**
  ({-1, 0, +1}).

`Bonsai-1.7B-unpacked` (used in the original version of this document) is the
**binary** variant. A binary-QAT model's weights have no training pressure toward a
zero state at all, so of course a ternary derivation applied to it showed near-0%
sparsity - not because of some exotic training-recipe difference, but because it
genuinely isn't a ternary model. The "2x scale factor" and "different sparsity-
inducing pressure" hypotheses in the original version were solving a problem that
didn't exist.

This was caught only after being asked directly which specific repo had been used,
and confirmed by checking sibling packed-variant names on the HF Hub - a check that
should have been done before treating the sparsity gap as a genuine research finding.
Everything below is redone against the correct `prism-ml/Ternary-Bonsai-1.7B-unpacked`.

---

## 1. Architecture match (confirmed, not assumed)

`prism-ml/Ternary-Bonsai-1.7B-unpacked` and `Qwen/Qwen3-1.7B` are the identical
`Qwen3ForCausalLM` class - 28 layers, hidden size 2048, 16 attention heads / 8 KV
heads - differing only in `vocab_size` (151669 vs. 151936). Re-confirmed directly via
a real `compare_qat_weights.py` run against the corrected repo: the only skipped
(shape-mismatched) parameter was `embed_tokens.weight` (151936x2048 vs.
151669x2048), exactly as expected - every transformer-block weight matched and was
compared.

## 2. The published Ternary-Bonsai weights are genuinely, cleanly ternary (group_size=128, row-basis)

Direct inspection of a real `model.layers.0.self_attn.q_proj` weight (shape
2048x2048) from the corrected repo:

- **36.6% exact-zero entries** (1,536,096 / 4,194,304).
- **16 distinct nonzero magnitudes** in a single output row - at first this looked
  like it might mean these are still continuous QAT master weights rather than a
  clean ternary code (a naive *per-tensor* or *per-row* single-scale ternary weight
  would show only 1 nonzero magnitude per row, not 16). That reading was wrong: `16 =
  2048 / 128` exactly - Bonsai's real training-time quantization uses group_size=128
  on a row basis (confirmed directly: splitting row 0 into its 16 groups of 128
  columns, EVERY group has exactly 1 distinct nonzero magnitude - a perfectly clean
  ternary code, just scaled per 128-column group rather than per whole row/tensor).
  This is genuinely, textbook-cleanly ternary - not an in-between "mostly continuous
  with a zero cluster" case as the previous version of this section claimed.

`get_ternary_code()`'s auto-detection (checks for a real exact-zero cluster before
deciding whether to use `torch.sign()` directly or derive via absmean+round+clamp)
correctly detects this and uses `torch.sign()` directly - and this is not a lucky
coincidence: since the underlying values are already genuinely ternary, `sign()` is
the exactly correct code regardless of what per-group scale multiplies each group
(sign is invariant to any positive per-group scale), so this auto-detection is
robust to the specific grouping scheme without needing to know it in advance.

## 3. What ternarization actually does to Qwen3-1.7B's weights (real, corrected numbers)

Comparing the (now correctly derived) ternary code against the real
`Qwen/Qwen3-1.7B` base weights, across all 28 layers:

| Layer type | Sparsity | Sign agreement | Kept/zeroed \|base\| ratio | Row-sparsity vs. row-norm correlation | Col-sparsity vs. col-norm correlation |
|---|---|---|---|---|---|
| Attention (QKVO) | 39.0% | 0.866 | 1.87 | -0.119 | -0.557 |
| MLP (gate/up/down) | 40.4% | 0.869 | 1.80 | -0.498 | -0.620 |

Reading these:

- **Sparsity (~39-40%)** is now closely in line with the three other ternary/BitNet
  checkpoints (see section 4) - no anomaly.
- **Sign agreement (0.866-0.869, well above the 0.5 chance level and notably
  HIGHER than the 0.68-0.75 mistakenly attributed to the binary checkpoint)**:
  ternarization strongly preserves the direction of the original Qwen3 weight where
  it keeps a nonzero value - a genuine ternary QAT process tracks the base weight's
  sign considerably more faithfully than the binary-model numbers had suggested.
- **Kept/zeroed |base| ratio > 1 in both layer types (1.87 attention, 1.80 MLP)**:
  ternarization behaves like magnitude-based pruning - entries that end up nonzero
  tend to have larger |Qwen3-weight| than entries that end up zero, similarly
  strong in both layer types (unlike the binary-checkpoint numbers, which showed a
  much bigger attention-vs-MLP gap).
- **Negative row/col-sparsity-vs-norm correlation, notably stronger for columns
  (-0.56 to -0.62) than rows (-0.12 to -0.50)**: input channels (columns) with
  larger overall weight norm are markedly less likely to be zeroed than output
  channels (rows) with large norm - a real structural asymmetry worth further
  investigation (not present at this strength in the earlier, incorrect analysis).

## 4. Cross-family ternary comparison: Bonsai now matches the other three closely

`compare_ternary_models.py`, corrected repo, all four models on MLP `down_proj`:

| Model | Sparsity | Sign balance | Row-sparsity std | Col-sparsity std |
|---|---|---|---|---|
| prism-ml/Ternary-Bonsai-1.7B-unpacked | 40.8% | ~0.00 | 0.0157 | 0.0528 |
| tiiuae/Falcon-E-1B-Base | 41.4% | ~0.00 | 0.0225 | 0.0436 |
| microsoft/bitnet-b1.58-2B-4T | 41.3% | ~0.00 | 0.0250 | 0.1108 |
| 1bitLLM/bitnet_b1_58-3B | 35.2% | ~0.00 | 0.0186 | 0.0388 |

All four checkpoints now cluster tightly: sparsity within a 6.2-percentage-point band
(35.2-41.4%), row-sparsity std within the same order of magnitude (0.0157-0.0250),
all sign-balanced (~0.00, no systematic +1/-1 bias). **No outlier, no anomaly.** The
originally-reported "0.8% sparsity, 2.5-3x higher row-variance" result for Bonsai was
entirely an artifact of the wrong checkpoint (section: ERRATUM above) - there is no
evidence here of a genuinely different ternarization recipe across these four
independently-built checkpoints; if anything, the tight clustering across two
architecture families (Qwen3/Llama-style vs. BitNet-style) and three organizations
is itself a real, mildly interesting finding - ternary QAT seems to converge to a
broadly similar sparsity/structure regardless of base architecture, at least for
these four checkpoints.

## 5. Preprocessing summary for this specific pair

- `Qwen/Qwen3-1.7B`: loads directly via `AutoModelForCausalLM.from_pretrained` - no
  quantization-related preprocessing needed at all, plain float weights.
- `prism-ml/Ternary-Bonsai-1.7B-unpacked`: also loads directly via
  `AutoModelForCausalLM.from_pretrained` (no unpacking step needed - it's
  `float_linear` format, not packed). Its weights are genuinely, cleanly ternary
  with group_size=128 quantization on a row basis (confirmed: every 128-column group
  within a row has exactly 1 distinct nonzero magnitude) - `torch.sign()` gives the
  exactly correct ternary code regardless of not knowing the group_size in advance,
  since sign is invariant to any positive per-group scale.
  `compare_ternary_models.py`'s auto-detecting `get_ternary_code()` uses this path
  directly; `compare_qat_weights.py`'s `ternarize()` instead always applies a
  per-tensor absmean derivation (no auto-detection there), which still gives sensible,
  consistent sparsity numbers on this checkpoint since the real zero mass dominates
  either way.
  **The one critical preprocessing step for this whole model family is simply using
  the correct repo** - `Ternary-Bonsai-*`, not `Bonsai-*` (binary) - confirmed by
  checking sibling packed-variant names (`*-mlx-2bit` vs. `*-mlx-1bit`) on the HF Hub,
  not by trusting the base repo name alone.
