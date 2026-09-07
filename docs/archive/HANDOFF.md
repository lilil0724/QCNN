# Handoff notes

This file exists so a fresh session/agent (potentially a different model) can pick up
this project without needing any prior conversation context. It covers: what's been
done, what's here vs. what's deliberately not, and what else exists in the source
repo (`QuantizedASR`) that could be relevant later.

## Current state

- `tools/compare_qat_weights.py` compares `Qwen/Qwen3-1.7B` (base) against
  `prism-ml/Ternary-Bonsai-1.7B-unpacked` (same architecture, ternary-QAT) on a
  **static weight basis only**: per-layer sparsity, sign agreement (does ternary QAT
  preserve the base model's sign where it keeps a value nonzero?), a kept-vs-zeroed
  `|base weight|` ratio (does it behave like magnitude pruning?), and per-row/per-column
  sparsity vs. norm correlation. Aggregated by layer type (attention QKVO vs. MLP).
  Real-GPU-verified in this repo's own location on 2026-07-18 - see `README.md` for
  the exact command and `docs/BONSAI_QWEN3_1.7B_FINDINGS.md` for prior numeric
  results (from when this tool lived in `QuantizedASR`).
- **This repo does NOT yet do any forward-pass/activation-based comparison of the
  two models** - everything here is a static weight-tensor diff. No real-audio-or-
  text-calibration-data pass has been run through either model in this repo yet.
- `prism-ml` publishes Bonsai under two DIFFERENT repo-name prefixes with different
  quantization: `prism-ml/Bonsai-*` (binary, {-1,+1}) vs.
  `prism-ml/Ternary-Bonsai-*` (ternary, {-1,0,+1}). Read
  `docs/BONSAI_QWEN3_1.7B_FINDINGS.md`'s erratum before using any new checkpoint from
  this author - an earlier pass in the source repo used the wrong one and produced a
  nonsensical result before this was caught.

## Environment

- Remote GPU box: `ssh ubuntu@140.114.79.186`. This repo lives at
  `/media/samsung/projects/qwen3-ternary-bonsai-analysis`, remote-only (no local
  mirror currently exists - if one gets added later, keep it in sync with `md5sum`
  verification, not just an assertion that a copy succeeded).
- Conda env `asr` already has every dependency this repo needs (`torch`,
  `transformers`, `numpy`, `pandas`, `matplotlib` - see `requirements.txt`):
  `source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate asr`.
- Use the configured cache for every command that touches the HF Hub:
  `HF_HOME=/home/pcs5060ti/Desktop/hf
  HF_HUB_CACHE=/home/pcs5060ti/Desktop/hf/hub`. Both `Qwen/Qwen3-1.7B` and
  `prism-ml/Ternary-Bonsai-1.7B-unpacked` are already cached there as of 2026-07-18.
- Current extracted-pair root (updated 2026-08-30):
  `/home/pcs5060ti/Desktop/qcnn_data/pairs/`. New pair extraction, baseline,
  prediction-model training, assembly, and decomposition runs should use pair
  directories under this root. Older paths under `/hdd/edwin/qwen3vsbonsai/pairs/`
  in historical result notes describe where those runs originally lived; they are
  not the current default.
- The source repo, `QuantizedASR`, lives alongside this one at
  `/media/samsung/projects/QuantizedASR` - useful for cross-referencing history (see
  below) but this repo does not depend on it (no shared code, confirmed at copy time -
  `tools/model_loading_utils.py` is a full, standalone extraction of the two small
  helper functions actually needed).

## Relevant tools that exist in `QuantizedASR` but were NOT copied here

These were left in the source repo per an explicit "minimal first copy" scoping
decision (only `compare_qat_weights.py` + its direct dependency) - not because
they're irrelevant. If this project's scope grows, these are the most likely next
things to pull over (copy, don't move, matching how this repo itself was created):

- **`tools/analysis/compare_ternary_models.py`** - compares Bonsai against THREE
  other, architecturally-different ternary/BitNet checkpoints (`tiiuae/Falcon-E-1B/3B`,
  `microsoft/bitnet-b1.58-2B-4T`, `1bitLLM/bitnet_b1_58-3B`) via architecture-agnostic
  intrinsic statistics (sparsity, sign balance) - useful if this project ever wants to
  compare Bonsai's ternarization behavior against how OTHER ternary/BitNet models
  ternarize, not just against its own float baseline.
- **`tools/analysis/compare_weights_gw.py`** - Gromov-Wasserstein (shape/permutation-
  agnostic) weight comparison, built specifically because element-wise diff and CKA
  both fail for models with no shared lineage/embedding space. Less relevant here
  since Qwen3-vs-Bonsai DOES share exact layer shapes/lineage (element-wise diff, what
  `compare_qat_weights.py` already does, is the right tool for that) - but worth
  knowing about if this repo ever compares Bonsai against a differently-shaped model.
- **`tools/analysis/analyze_hqq_scales.py`** - structural analysis of HQQ's
  quantization scale/zero-point distributions (is log2(scale) roughly normal? entropy
  vs. bits-per-scale?) - the "item 4a" analytical-quantization-scheme sibling to this
  repo's own "item 4b" work. Not directly applicable to Bonsai (Bonsai isn't
  HQQ-quantized), but same research question/spirit if this project ever explores an
  analytical (rather than learned) ternarization scheme for Bonsai's own scales.
- **`tools/analysis/compute_cka.py`** - linear CKA between two models' REAL
  activations, layer by layer - answers "how similar are the two models' internal
  representations," which a static weight diff cannot show. **This is the most
  natural next tool to bring over** if/when this project wants to move beyond static
  weight comparison - real audio/text data through both Qwen3-1.7B and Ternary-Bonsai
  would show where their internal representations diverge, not just where their
  weights differ.
- **`tools/analysis/error_propagation.py`** - isolates quantization to ONE layer at a
  time (everything else float) and tracks how that layer's error compounds into LATER
  layers via CKA, across a real full-model forward pass. Built for PTQ (HQQ/quanto/
  bnb/torchao), not QAT - would need adaptation for a QAT-ternary comparison like
  Bonsai (there's no "everything else float, one layer ternary" middle state readily
  available the way there is for PTQ's `--quant_skip_modules`), but the underlying
  methodology (depth-vs-divergence curve) is directly relevant to "which of Bonsai's
  layers matter most."
- **Two BRAND NEW, generic methodologies built in `QuantizedASR` on 2026-07-18** (for
  an unrelated HQQ v1-vs-v2 investigation, but NOT HQQ-specific in design - both
  compare two same-architecture models' actual `forward()` behavior, not just static
  weights, which is exactly the gap noted above for this repo):
  - `hqq/tests/test_hqqv2_random_input_per_layer.py`'s pattern: for every matched
    layer, feed an identical random input tensor through each model's corresponding
    layer's real `forward()` and compare outputs (not just weights). Directly
    adaptable to Qwen3-vs-Bonsai: would show which layers' OUTPUT behavior differs
    most under synthetic input, complementing `compare_qat_weights.py`'s static view.
  - `tools/analysis/compare_ptq_weights.py`'s new `compare_isolated_layer_errors()`:
    for one real calibration sample, capture a layer's real INPUT (via
    `register_forward_pre_hook`) from one model, replay it through the OTHER model's
    corresponding layer in isolation, and compare against that layer's real output.
    Measures each layer's own marginal behavioral difference given a real, correct
    input - not yet adapted for a QAT (rather than PTQ) comparison, but the mechanism
    doesn't actually depend on PTQ specifically, just on having two loaded models with
    matched layer names (which Qwen3-1.7B/Ternary-Bonsai-1.7B-unpacked have).
- **`qasr/model/bitnet_convert.py`'s `load_and_convert_to_bitnet_linear()`** (NOT
  extracted into this repo's `model_loading_utils.py` - only the two smaller loading
  helpers were) - loads a model and swaps its `nn.Linear` layers for `onebitllms`'s
  trainable `BitNetLinear`. Relevant if this project's next phase is to actually
  CONTINUE TRAINING/further QAT-tune Bonsai (or reproduce its ternary-QAT recipe from
  scratch), not just statically analyze the published checkpoint.
- **`docs/session_notes/WEIGHT_ANALYSIS_METHODS_FINDINGS.md`** - a survey of weight-
  comparison methods (element-wise diff, CKA-on-weights, Gromov-Wasserstein) written
  specifically while building the tools above; explains why each tool was built the
  way it was and what alternatives were rejected and why. Worth reading in full before
  building any NEW weight-comparison tool for this project, to avoid re-deriving
  conclusions already reached there.
- **`docs/session_notes/CLAUDE_CHANGES.md`** (in `QuantizedASR`, not copied here) -
  the full dated history of every tool mentioned above, including real bugs found and
  fixed, real numbers from actual GPU runs, and the exact reasoning behind each design
  decision. Search it for "Bonsai" for the complete narrative if
  `docs/BONSAI_QWEN3_1.7B_FINDINGS.md` alone isn't enough context for a given question.

## Suggested next steps (not started, in rough priority order)

1. If continuing static weight analysis: nothing urgent - `compare_qat_weights.py`'s
   existing per-layer/per-layer-type report already covers sparsity, sign agreement,
   and magnitude-pruning-like behavior.
2. If moving to behavioral/functional comparison: port `compute_cka.py` (real
   activations, both models, real data) and/or adapt the two new forward-pass
   methodologies listed above - this would answer "do Qwen3 and Bonsai actually
   process the same input similarly, layer by layer," which nothing in this repo
   currently answers.
3. If comparing Bonsai against other ternary/BitNet families: copy over
   `compare_ternary_models.py` (and `compare_weights_gw.py` if a non-Qwen3-architecture
   ternary model gets added to the comparison set).
