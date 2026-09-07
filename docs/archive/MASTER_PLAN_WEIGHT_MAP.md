# MASTER PLAN: Learning the FP-to-Low-Bit Weight Map (Qwen3 -> Bonsai)

Status: proposal (nothing implemented yet). Written 2026-07-18 after reading every file
in this repo (including `quantization_cnn.md`, the ChatGPT survey) and running an
independent novelty/feasibility verification (searches + HF Hub checks + the real
findings in `docs/BONSAI_QWEN3_1.7B_FINDINGS.md`).

The idea under evaluation: train an auxiliary network (CNN encoder-decoder or similar)
that takes full-precision Qwen3 weights as input and predicts the corresponding
Bonsai binary/ternary QAT weights - a supervised weight-to-weight regression, where
the QAT checkpoints act as golden labels. Single-pass, diffusion-like, or
autoregressive variants were all floated; this plan takes a position on which to
build first and why.

---

## 0. VERDICT FIRST: publication potential, and why

**Workshop level (top-tier venue workshops): YES, with high confidence.**
**Main conference level: POSSIBLE but conditional** - it depends on results that
cannot be known in advance, and this plan is structured around explicit go/no-go
gates that tell us as early and cheaply as possible which tier we're in.

### Why yes (workshop, and the path to main conference)

1. **The formulation is genuinely novel.** My independent search (see section 1)
   confirms the ChatGPT survey's conclusion from a different direction: not only does
   no quantization method use a learned network over weight tensors, but the
   *weight-space learning* literature (metanetworks, hypernetworks, checkpoint
   generative models - a whole community the ChatGPT survey did not check) also has
   no published FP->QAT checkpoint translation. There is even a dedicated ICLR 2025
   Workshop on Weight Space Learning whose call-for-papers this project fits exactly.
   The closest precedents (MetaPruning/DHP predict pruned-model weights;
   HyperDreamBooth predicts personalization weights; G.pt/p-diff generate checkpoints
   with diffusion) establish that predicting derived-model weights *works as a
   paradigm* - none of them target quantization, none operate at LLM scale on real
   released checkpoint pairs.

2. **A unique, real dataset nobody else has assembled.** prism-ml publishes Bonsai in
   BOTH binary ({-1,+1}) and ternary ({-1,0,+1}) families, unpacked, at 1.7B / 4B /
   8B / 27B - and I verified via config.json that 1.7B/4B/8B are all plain
   `Qwen3ForCausalLM`, i.e. **six clean same-architecture (Qwen3-base -> QAT) pairs
   across three scales and two target formats**, plus a Qwen3.5-arch 27B pair and an
   image-model pair as stress tests. Each pair represents a genuinely expensive
   QAT+distillation run (prism-ml distills from the FP teacher with reasoning
   traces). This is far better than the "only 4 models" worry: at the sample level
   it's millions of 128-column weight groups; at the *recipe* level it's still
   effectively one vendor (the key limitation - see gate G3 and the mitigation in
   section 6.4).

3. **The scientific question is interesting even if the method loses.** "How much of
   ternary QAT's outcome is a deterministic, *local* function of the base weights?"
   is a real question about QAT itself, and this repo's own findings already bound it
   from below: sign agreement 0.866-0.869 and kept/zeroed |base| ratio ~1.8 mean a
   trivial magnitude-based rule already predicts most of the code. BitDelta (NeurIPS
   2024, "Your Fine-Tune May Only Be Worth One Bit") showed fine-tune deltas carry
   ~1 bit of information per weight; whether the QAT delta is similarly
   low-information *and locally predictable* is unknown, and either answer is a
   finding. A careful predictability-ceiling analysis with layer-type/depth breakdown
   is a solid workshop paper even in the pessimistic case.

### Why the main-conference bar is at risk (the honest part)

1. **The trivial baseline is strong and the headroom is thin.** Deriving the ternary
   code directly from the base weights (absmean ternarize, this repo's own
   `ternarize()`) already agrees with Bonsai's sign at ~87% of kept positions and
   reproduces the ~39-40% sparsity. The learnable signal is the residual ~10-15% of
   entries - and some unknown fraction of that residual is *irreducible noise*
   (seed/data-order dependence of the QAT run), which no network can predict. If the
   noise floor eats most of the residual, the method caps out as "slightly better
   than absmean," which is not a main-conference result. We can't measure the noise
   floor from Bonsai alone (prism-ml ran QAT once per model); section 6.4's own-QAT
   multi-seed experiment is designed to measure it directly at small scale.

2. **Weight-space accuracy is not the metric that matters, and the functional metric
   is unforgiving.** A predicted ternary model must be evaluated by
   perplexity/zero-shot accuracy against (a) the real Bonsai (upper bound), (b) naive
   ternarization of Qwen3 (lower bound), and (c) strong calibration-based PTQ
   binarization baselines (BiLLM, ARB-LLM, OneBit-style init). Beating (c) without
   using calibration data is the result that would carry a main-conference paper.
   It is entirely possible to improve code accuracy from 87% to 93% and see almost
   no functional gain - or even see the biggest functional lever be the *scales*,
   not the code.

3. **One vendor, one recipe.** Every available pair comes from prism-ml's single QAT
   recipe. Falcon-E and Microsoft BitNet have no FP base counterpart (trained from
   scratch), so cross-recipe generalization cannot be tested on public checkpoints.
   Reviewers will ask. The mitigation (own small-scale QAT pairs at 0.6B, multiple
   seeds, via onebitllms - the machinery already exists in `QuantizedASR`'s
   `bitnet_convert.py`) is essential for the main-conference version, not optional.

4. **The "weights are images" intuition is partially wrong, and the plan must say
   so.** In a transformer linear layer, adjacent *rows* (output neurons) are an
   arbitrary ordering - there is no translation-invariant structure across rows the
   way there is across image pixels. What IS real structure: (a) the 128-column
   quantization groups along the input axis, (b) per-column (input-channel)
   statistics that are shared across rows - and this repo's own finding of a strong
   column-sparsity-vs-norm correlation (-0.56 to -0.62, much stronger than rows)
   confirms cross-row column structure is informative. So locality along the
   *column/group axis* is defensible; locality across rows is not. The architecture
   and ablations below are designed around that asymmetry, and "CNN vs
   context-feature MLP" is a headline ablation, not a footnote - if a per-element MLP
   with handcrafted row/col/group context features matches the CNN, the CNN story
   dies and the paper becomes about *what information* predicts QAT, which is still
   publishable but different.

**Bottom line**: proceed. Structure the work so that after ~2 weeks (gate G1) we know
the baseline headroom, after ~5 weeks (gate G2) we know whether learning beats the
trivial rule on held-out layers, and only then invest in the expensive
functional/cross-scale/own-QAT experiments that decide workshop vs main conference.

---

## 1. Novelty verification (my own search, 2026-07-18)

The ChatGPT survey (`quantization_cnn.md`) covered the *quantization-methods*
literature well (AdaRound, SmoothQuant, QuaRot, SpinQuant, AWQ, OmniQuant, LLM-QAT,
LR-QAT, BitNet, GPTQ, D2Quant) and correctly concluded none of them applies a learned
network over weight tensors. I verified its gap claim from three directions it did
not check:

1. **Weight-space learning / metanetworks** (the community whose core object IS
   networks-that-eat-weights): graph metanetworks, neural functionals (NFN, DWSNets),
   hyper-representations, G.pt ("Learning to Learn with Generative Models of Neural
   Network Checkpoints"), p-diff ("Neural Network Diffusion"), geometric flow models
   over weights (2025). All do property prediction, generation, or optimization -
   **none do supervised FP->QAT checkpoint translation**. The ICLR 2025 Weight Space
   Learning workshop's scope ("model synthesis", "weights as a data modality")
   fits this project exactly - both a novelty confirmation and a natural venue.

2. **Hypernetworks predicting derived-model weights**: MetaPruning (ICCV'19) and DHP
   (ECCV'20) predict weights of *pruned* architectures; HyperDreamBooth (2023)
   predicts personalization (LoRA-like) weights from an image. These are the closest
   methodological ancestors and should be cited as such - but they are
   vision-scale, pruning/personalization-targeted, and none takes the FP weight
   tensor itself as the input to be *translated* to a QAT outcome.

3. **Low-bit LLM conversion methods** (what a practitioner would actually use today
   to get a binary/ternary LLM from an FP one): BiLLM (1-bit PTQ), ARB-LLM
   (alternating refined binarization), OneBit (SVD-based init + KD training),
   BitDistiller (self-distillation QAT), SDQ-LLM, LBLLM (three-stage distillation,
   2026). All are per-model optimization/training procedures with calibration data
   or training tokens; **none amortizes the conversion into a reusable learned
   weight-map trained on released (FP, QAT) checkpoint pairs**. These are the
   baselines, not the prior art for the core idea.

Also directly relevant as motivation: **BitDelta** (NeurIPS 2024) - fine-tune deltas
compress to 1 bit/weight, i.e. checkpoint-to-derived-checkpoint deltas are
low-information. The Bonsai delta may be similar; that is the learnability bet made
explicit.

**Conclusion: the specific formulation - a reusable network trained by supervised
regression/classification on real released (FP base, QAT low-bit) LLM checkpoint
pairs - appears unpublished as of 2026-07.** The idea sits at the intersection of
three literatures that each stop one step short of it.

One caveat for the paper's framing: the novelty is the *formulation and the study of
QAT predictability*, not "CNNs applied to weights" per se (metanetworks already eat
weights). Do not oversell the architecture; sell the question and the dataset.

---

## 2. Data inventory (verified on HF Hub, 2026-07-18)

| Pair | Base (public) | QAT target | Arch | Status |
|---|---|---|---|---|
| 1 | Qwen/Qwen3-1.7B | Ternary-Bonsai-1.7B-unpacked | Qwen3, 28L, h2048 | verified, both cached on box |
| 2 | Qwen/Qwen3-1.7B | Bonsai-1.7B-unpacked (binary) | Qwen3 | repo exists; binary confirmed via mlx-1bit sibling |
| 3 | Qwen/Qwen3-4B | Ternary-Bonsai-4B-unpacked | Qwen3, 36L, h2560 (verified) | config verified |
| 4 | Qwen/Qwen3-4B | Bonsai-4B-unpacked | Qwen3 | repo exists |
| 5 | Qwen/Qwen3-8B | Ternary-Bonsai-8B-unpacked | Qwen3, 36L, h4096 (verified) | config verified |
| 6 | Qwen/Qwen3-8B | Bonsai-8B-unpacked | Qwen3 | repo exists |
| 7 | Qwen3.6-27B (per press) | (Ternary-)Bonsai-27B-unpacked | Qwen3_5ForConditionalGeneration | held-out arch stress test; verify base availability first |
| 8 | TBD | bonsai-image-(binary/ternary)-4B | image model | unpacked config 404'd on first check; investigate later, skip for PoC |

To verify before relying on pairs 2-6: (a) each Bonsai checkpoint really is
initialized from the matching Qwen3 base (run `tools/compare_qat_weights.py` per pair;
sign agreement >> 0.5 is the fingerprint - pair 1 already confirmed at 0.87);
(b) each `Ternary-*-unpacked` is cleanly ternary per 128-group and each `Bonsai-*`
is cleanly binary (extend the group-structure check from the findings doc).
**Lesson from the erratum applies with force here: never mix up the two families.**
The family membership of every checkpoint used must be asserted in code
(`assert_quantization_family()`, section 5) - not trusted from the repo name.

Effective dataset size: pair 1 alone has ~196 transformer-block matrices /
~1.4B weight elements / ~11M ternary 128-groups. Pairs 1-6 together: ~5x that.
Samples are abundant; *recipes* are scarce (n=1 vendor). Splits must therefore be
by layer/depth/model-scale, never by random element, and every claim must be
phrased per-recipe until section 6.4's own-QAT pairs add recipe diversity.

Storage: bases + unpacked QAT checkpoints for 1.7B/4B/8B ~ (3.4+8+16)*2 GB ~ 55 GB
in bf16 on `/home/pcs5060ti/Desktop/hf` (`/media/samsung` is 97% full,
put nothing large there). Extracted patch tensors: store as memory-mapped safetensors
shards under `/hdd/edwin/qwen3vsbonsai/patches/`.

Hardware reality: one RTX 3090 (24 GB). The weight-map itself is tiny (<10M params)
and trains on patches - single-GPU friendly. Functional eval of assembled 1.7B/4B
models fits easily; 8B fits in bf16 (16 GB) with care; 27B only via offload/gguf
(stretch). Own-QAT runs (section 6.4) at 0.6B are the practical scale for this box.

---

## 3. Problem formulation (and three corrections to the ChatGPT sketch)

Learn f_theta mapping a base-weight block plus context to the QAT outcome for the
matched block. Two heads:

- **Code head**: per-element 3-way classification over {-1, 0, +1} (ternary) or
  2-way (binary). **Correction 1 to the ChatGPT sketch: this is classification, not
  MSE regression.** The published Bonsai weights are exactly ternary-code x per-group
  scale; MSE-on-fake-quantized-weights (the sketch's loss) mixes code errors with
  scale errors, has degenerate gradients near boundaries via STE, and optimizes a
  proxy that isn't the eval metric. Cross-entropy on the code, with the {-1,+1}
  classes reweighted vs the ~40% zero class, is cleaner, and the softmax probabilities
  give calibrated uncertainty (useful for the "which entries are unpredictable"
  analysis). Keep an STE-MSE variant only as an ablation arm.
- **Scale head**: per-128-group scale regression, Huber loss on log(scale), predicted
  as a correction to the absmean scale of the base group (residual parameterization -
  that part of the sketch is right and stays).

**Correction 2: residual-from-baseline, made explicit.** The network predicts a
*deviation from the absmean-ternarized base code*, not the code from scratch - e.g.
logits initialized/biased so that zero network output reproduces the absmean
baseline exactly. This guarantees the learned model starts AT the strong trivial
baseline and can only be judged by what it adds. (The sketch's residual is in weight
space; ours is in decision space, which is what actually gets evaluated.)

**Correction 3: the input representation must respect the real symmetry structure,
not image locality.** Per section 0.4: rows are permutable, columns carry shared
meaning, groups are the quantization unit. Input per element: the base weight
normalized by its group absmean, plus context features (group statistics, row
statistics, column statistics, |w| rank within group) and conditioning embeddings
(layer type QKVO-vs-MLP-projection identity, depth fraction, target family
binary/ternary, model scale).

Single-pass vs diffusion vs autoregressive: **single-pass first, and probably only.**
The output space per group is small (129 elements x 3 classes with one shared scale);
the map is plausibly near-deterministic given context (that's the research bet); and
both diffusion and AR variants pay large costs (sampling steps / sequence length ~
millions of elements) to model *distributional multimodality* we have no evidence of
yet. The right place for a diffusion-like mechanism is a second iteration IF the
single-pass model's errors show correlated, multimodal structure (e.g. whole groups
flipping together between plausible configurations). Note it as future work; do not
build it first. (Precedent for the AR/diffusion routes existing at all: G.pt, p-diff -
both operate on tiny networks, which is exactly why single-pass is the scalable choice
here.)

### Candidate architectures (all small, <10M params)

- **A0 - ContextMLP (the null-hypothesis model)**: per-element MLP on the handcrafted
  context features above. No locality at all. If nothing beats this, the finding is
  "QAT predictability is carried by simple statistics" - report it honestly.
- **A1 - GroupConv1D**: 1D CNN along the column axis within a row (group-aligned,
  kernel sizes spanning within-group and cross-group context), per-row independent,
  plus pooled column-statistics context broadcast across rows. Respects the real
  structure; this is the "CNN with the right inductive bias" candidate.
- **A2 - 2D residual CNN / small UNet on (rows x cols) patches** (e.g. 128x512
  patches): the literal ChatGPT proposal. Included mainly so the row-locality
  assumption gets tested rather than assumed.
- **A3 (later, only if A1 wins and needs more context)**: axial attention across the
  row axis (permutation-equivariant, unlike row-convolution) on top of A1.

---

## 4. Evaluation protocol

### 4.1 Weight-space (cheap, runs constantly)
- Per-element code accuracy + macro-F1 over {-1,0,+1}, overall and on the *hard
  subset* (entries where absmean baseline disagrees with Bonsai - the only entries
  that matter; ~10-15% per the findings).
- Per-group scale relative error.
- Breakdowns: layer type x depth x scale. (The findings' column-norm correlation
  says structure varies by axis; show it.)
- Baselines: B0 absmean per-tensor (this repo's `ternarize()`), B1 absmean per-group-128,
  B2 magnitude-threshold tuned per layer type on train split, B3 logistic regression
  on the same context features (sanity: is the MLP even needed?).

### 4.2 Functional (the metric that decides the paper)
Assemble a full model: take the real Bonsai checkpoint, replace transformer-block
linears with predicted code x predicted scales (embeddings/lm_head/norms kept from
Bonsai - vocab differs from base anyway, 151669 vs 151936). Then:
- Perplexity: WikiText-2, C4 slice.
- Zero-shot: lm-eval-harness (ARC-e/c, HellaSwag, PIQA, WinoGrande, BoolQ).
- KL divergence of logits vs real Bonsai on a fixed prompt set (are we recovering
  *Bonsai*, or just *a* working ternary model? both are interesting, distinguish them).
- **Oracle decompositions** (do these FIRST, they bound everything): (real code, real
  scales) = Bonsai itself; (real code, absmean scales) isolates how much scales
  matter; (absmean code, real scales) isolates how much code matters; then predicted
  variants. If (real code, absmean scales) already collapses, scale prediction is
  the whole game and the code head is a sideshow - better to learn that in week 3
  than at review time.
- Comparison points: real Bonsai (upper), B0/B1-assembled (lower), BiLLM / ARB-LLM /
  OneBit-init reproductions on Qwen3-1.7B (the competitive bar; they use calibration
  data, we don't - a fair asymmetry to highlight either way).

### 4.3 Generalization ladder (each rung a separate claim)
1. Held-out layers within a model (interpolation).
2. Held-out depth range (e.g. train on layers 0-20, test 21-27).
3. Held-out scale: train on 1.7B+4B pairs, test on 8B. **This is the headline
   generalization experiment** - it's what makes the method "reusable" rather than
   a per-model fit.
4. Cross-family: train ternary, test binary (and joint with family conditioning).
5. Cross-arch stress: Qwen3.5-27B pair, image pair (expected to degrade; report).
6. Cross-recipe (needs 6.4's own pairs): the make-or-break external-validity rung.

---

## 5. Proof-of-concept: code to add (all new files; nothing existing is touched)

New package `weight2ternary/` in this repo, with `tests/` runnable via
`python -m pytest tests/ -x -q` in the `asr` conda env (torch/transformers/numpy/
pandas already there; add `pytest`, `safetensors`, `datasets`, `lm-eval` to
requirements as a new `requirements-w2t.txt` rather than editing `requirements.txt`).

```
weight2ternary/
  __init__.py
  family_check.py     assert_quantization_family(repo_id) -> 'binary'|'ternary'
                      - downloads ONE shard, checks exact-zero fraction and per-128-group
                        distinct-magnitude structure; hard-fails on mismatch with the
                        repo-name prefix. Codifies the erratum lesson as an assertion.
  extract.py          iter_matched_blocks(base_id, qat_id, split_spec) -> yields
                      (base_block, code_block, scale_block, context) from safetensors
                      WITHOUT loading full models (lazy per-tensor loading);
                      derive_code_and_scales(qat_w, group=128) - recovers exact ternary
                      code + per-group scales from unpacked weights, asserting exact
                      reconstruction (this is the ground-truth extractor; it must be
                      bit-perfect, tested against pair 1 where the structure is known).
  features.py         per-element/group/row/col context features + conditioning encodings.
  baselines.py        B0/B1/B2/B3 (section 4.1) + code_accuracy/macro-F1/hard-subset
                      metrics. B0 must numerically match tools/compare_qat_weights.py's
                      ternarize() (cross-check in tests, do not import it - keep the
                      package standalone, matching this repo's copy-don't-couple style).
  models.py           ContextMLP (A0), GroupConv1D (A1), ResCNN2D (A2); all with the
                      decision-space residual parameterization (zero-init final layer
                      == absmean baseline exactly - unit-tested property).
  losses.py           class-weighted CE (code), Huber-on-log (scales), STE-MSE ablation arm.
  train.py            patch-sampler DataLoader over extracted shards, AdamW, cosine LR,
                      per-epoch weight-space eval on held-out layers, CSV logs under
                      results/ (gitignored already).
  evaluate.py         full weight-space report (section 4.1 tables) for a trained model
                      or baseline, per layer-type x depth.
  assemble.py         build_predicted_checkpoint(bonsai_id, predictions) -> HF model dir
                      (swap transformer-block linears only); includes the oracle modes
                      (real-code x absmean-scales etc.).
  eval_functional.py  perplexity (WikiText-2/C4) + lm-eval hooks + logit-KL vs Bonsai.
  augment.py          paired row/col permutations, positive per-group rescaling,
                      GQA-consistent head permutations (section 6.1).

tests/
  test_family_check.py   synthetic binary/ternary/continuous tensors -> correct family;
                         mismatch raises.
  test_extract.py        synthetic group-scaled ternary tensor -> derive_code_and_scales
                         reconstructs exactly; split spec leaks no layer across splits.
  test_baselines.py      B0 == compare_qat_weights.ternarize() on random tensors;
                         known-accuracy on synthetic magnitude-pruned pairs (reuse the
                         style of compare_qat_weights.py's _synthetic_self_test).
  test_models.py         zero-init == baseline exactly; shapes; A0/A1/A2 overfit a tiny
                         synthetic batch to ~100% (learnability smoke test);
                         A1 row-permutation equivariance.
  test_augment.py        each augmentation maps valid (input,target) pairs to valid pairs
                         (code/scale consistency preserved).
  test_assemble.py       oracle roundtrip: assemble(real code, real scales) == Bonsai
                         weights bit-exactly on one real layer (GPU box only; skipped
                         locally via marker).
```

PoC sequence (strictly ordered, each step gated on the previous):
1. `family_check` + `extract` + tests -> extract pair 1 to shards. (~2 days)
2. `baselines` -> **the week-1 number that gates everything: B0/B1/B2 code accuracy
   on pair 1, overall and hard-subset** (gate G1 below). (~2 days)
3. `models` + `train` -> A0 and A1 on pair-1 train split, held-out-layer eval (G2). (~1 wk)
4. `assemble` + oracle decompositions + functional eval of best model at 1.7B. (~1 wk)

---

## 6. Iterative extensions (post-PoC, priority order)

### 6.1 Weight augmentations (addresses "few models" at the sample level)
- **Paired row permutations** (and GQA-consistent head permutations for QKVO): the
  target co-permutes exactly, so these are label-preserving; they enforce the
  permutation-equivariance prior the true QAT map approximately has. NOTE for A2
  specifically this is also a falsification probe: a 2D CNN that relies on row
  adjacency will get *hurt* by row-permutation augmentation - that's the
  locality-assumption test running for free inside training.
- **Positive per-group rescaling** of the base input (code target invariant, scale
  target co-scales) - enforces the scale invariance `get_ternary_code()` exploits.
- **Small additive noise** on base weights (denoising-autoencoder flavor, the one
  ChatGPT extension worth keeping early).
- What augmentation CANNOT do, stated honestly: it adds invariance, not recipe
  diversity. It will not fix gate G3.

### 6.2 Ablations
- A0 vs A1 vs A2 (the inductive-bias question - a core result, see 0.4).
- Context ablation within A0: which features carry predictability (|w| rank in group
  alone? column norms? depth?). This doubles as the paper's analysis section.
- Loss: CE vs STE-MSE; class weighting on/off.
- Conditioning: with/without layer-type, depth, family, scale embeddings.
- Data scaling: accuracy vs #training layers/pairs (does pair 3-6 data help pair-1
  held-out layers? - amortization evidence).
- Augmentations on/off (6.1).

### 6.3 The two experiments that would carry a main-conference submission
- **Cross-scale transfer** (train 1.7B+4B -> predict 8B, functional eval).
- **Predicted-weights-as-QAT-init**: continue real ternary QAT (onebitllms
  BitNetLinear; loading machinery exists in QuantizedASR's `bitnet_convert.py` -
  copy, don't import) from (a) absmean init, (b) our predicted init, (c) random-code
  init; measure tokens-to-reach-target-perplexity. Even a modest "predicted init
  saves 30-50% of QAT tokens at 0.6-1.7B scale" is a compelling, practical claim.
  On one 3090 this is realistic only at 0.6B-1.7B with ~100-500M tokens - frame as
  demonstration, not production.

### 6.4 Own QAT pairs (fixes the two deepest validity holes)
Run ternary QAT on Qwen3-0.6B (fits the 3090) with onebitllms, 2-3 seeds, one or two
data mixes ->
- **Noise floor measured directly**: agreement between two same-recipe different-seed
  QAT runs is the predictability CEILING no model can exceed. Every accuracy number
  in the paper gets reported against it. (Without this, "87% -> 93%" is
  uninterpretable - 93% might be above or below what a re-run of QAT itself achieves.)
- **Recipe diversity**: our recipe != prism-ml's recipe -> the only available
  cross-recipe generalization test (ladder rung 6).
This is the single highest-value addition beyond the PoC. Budget ~2-3 weeks of
intermittent 3090 time.

### 6.5 Datasets/models to try, in order
1.7B ternary (PoC) -> 1.7B binary (family transfer) -> 4B, 8B (scale ladder) ->
own 0.6B pairs (noise floor + recipe) -> 27B Qwen3.5 + image pair (stress, optional).
Functional eval datasets: WikiText-2, C4, lm-eval suite (section 4.2); QAT-init
continuation data: a small open pretraining mix (e.g. FineWeb-Edu slice, cached to
/hdd/edwin - never the home dir).

---

## 7. Go/no-go gates

- **G1 (end week 1)**: B0/B1/B2 hard-subset size and structure on pair 1. If the
  absmean baseline is already >97% accurate overall AND oracle decomposition shows
  code errors barely matter functionally, the *learning* project dies early and
  cheaply; pivot to the scale/structure analysis paper. (Expected from the findings:
  ~85-90% overall accuracy, real headroom exists - but verify, don't assume.)
- **G2 (end week ~4)**: does any of A0/A1 beat B2 on hard-subset accuracy on held-out
  layers by a clear margin (>5 points)? No -> the residual is mostly noise/global;
  report the predictability-ceiling analysis as a workshop paper and stop there.
- **G3 (before main-conference commitment)**: does 6.4's cross-recipe test show
  transfer meaningfully above baseline? No -> the method is "per-recipe amortization",
  scope claims accordingly (still workshop-strong, main-conference-weak).

## 8. Timeline (single RTX 3090, one person, part-time-realistic)

| Weeks | Work | Output |
|---|---|---|
| 1-2 | PoC steps 1-2: extraction, family checks, baselines, tests | G1 numbers |
| 3-4 | PoC step 3: A0/A1 training + held-out-layer eval | G2 decision |
| 5-6 | PoC step 4: assembly, oracle decompositions, functional eval @1.7B | first functional table |
| 7-8 | Binary family + 4B/8B extraction; cross-scale transfer | generalization ladder rungs 1-4 |
| 9-11 | Own 0.6B QAT pairs (2-3 seeds) - runs in background of: | noise floor + G3 |
| 9-11 | ablations (6.2), BiLLM/ARB-LLM baseline reproductions | comparison table |
| 12-13 | writing, workshop submission | **workshop paper** |
| 14-20 | (conditional on G2+G3) QAT-init experiment, 27B/image stress, polish | **main-conference attempt** |

Total: ~3 months to a workshop-grade result, ~5 months to a credible main-conference
submission, with early-exit ramps at G1/G2 that cost only 1 and 4 weeks respectively
if the idea is empirically dead.

## 9. Expected outcomes & failure modes, stated in advance

- **Pessimistic (G2 fails)**: "Ternary QAT's deviation from magnitude-based
  ternarization is largely unpredictable from base weights; predictability ceiling X%
  measured via multi-seed QAT" + layer-type/depth structure analysis. Workshop paper
  (weight-space-learning or efficient-ML workshop), genuinely useful negative result.
- **Base case (G2 passes, functional gains modest)**: learned map beats naive
  ternarization functionally, approaches calibration-based PTQ without calibration
  data, transfers across scale within the recipe. Strong workshop paper / borderline
  main conference depending on margins.
- **Optimistic (G2+G3 pass + QAT-init works)**: "QAT outcomes are substantially
  predictable; a reusable weight-map cuts QAT cost by N% and transfers across scales
  and recipes." Main-conference submission with a practical hook and a scientific
  finding.
- **Known threats not yet mitigated**: Bonsai license/terms for derivative analysis
  (check before publishing); possibility that Bonsai's released "unpacked" weights
  went through post-QAT processing that decouples them from the base (the 0.87 sign
  agreement argues against, but check 4B/8B too); lm-eval + 8B functional runs are
  slow on one 3090 (batch/sequence discipline needed).

---

## 10. PoC EXECUTION STATUS (added 2026-07-18, after real runs)

PoC steps 1-3 are implemented and run for real on pair 1. Code: `weight2ternary/`
package (data_utils/model_utils/eval_utils), CLI tools `tools/extract_pair.py`,
`tools/run_baselines.py`, `tools/train_weight_map.py`, plain-assert tests under
`tests/` + `scripts/run_tests.sh` (all passing on the box, including a learnability
smoke test). Extracted pair shards:
`/hdd/edwin/qwen3vsbonsai/pairs/Qwen_Qwen3-1.7B_prism-ml_Ternary-Bonsai-1.7B-unpacked`
(196 layers, 28 depths). Logs/CSVs under `results/` (serials 1-4).

### Gate G1 - RESULT: passes, in the OPPOSITE direction from the section-0 worry

The findings doc's 0.87 "sign agreement" was conditional on Bonsai's KEPT entries;
full 3-way code agreement is far lower. On held-out layers (depth % 4 == 3):

| baseline | overall acc | hard_frac | hard acc |
|---|---|---|---|
| B0 per-tensor absmean | 0.606 | 0.394 | 0.054 |
| B1 per-group absmean  | 0.606 | 0.394 | 0 (by constr.) |
| B2 group threshold, tau=0.775 (tuned) | 0.618 | 0.394 | 0.193 |

The magnitude rule disagrees with Bonsai's real code on ~39% of entries, and the
zero-pattern IoU between the absmean rule and the real code is only ~0.31: WHICH
entries ternary QAT zeroes is largely not a magnitude decision. Headroom is huge -
the section-0 "thin residual" worry is dead; the live question became "is that 39%
predictable at all?"

(Extraction note: real Ternary-Bonsai weights are clean group-128 codes up to
bf16-ulp noise - max 0.73% relative within-group spread on a ~1e-4 fraction of
groups - so ground-truth scales are per-group nonzero-magnitude MEDIANS with a 2%
structural tolerance. The first strict-exactness extractor version false-rejected
on this; fixed and re-verified. Layer-0 q_proj cross-checks against the findings
doc: sparsity 0.366, kept-sign agreement 0.844.)

### Gate G2 - FIRST ANSWER: mostly NOT predictable from local context (so far)

Two miscalibrated runs (serials 1/2: --hard_weight 4.0, selection by hard acc)
reached hard-subset accuracy 0.73 - but by over-firing on easy entries, collapsing
OVERALL accuracy to 0.45 < 0.61 baseline. Recalibrated (serials 3/4: plain CE,
selection by overall accuracy):

| model | overall acc (val) | hard acc | scale relerr |
|---|---|---|---|
| A0 ContextMLP 0.04M | 0.623 | 0.187 | 0.128 |
| A1 GroupConv1D 0.51M | 0.624 | 0.194 | 0.127 |
| best rule (B2 tuned) | 0.618 | 0.193 | 0.193 (B1-derived scales) |

Calibrated learned models beat the best tuned rule by only ~0.6pp overall (and match
it on the hard subset); A1's column-axis convolutions add ~nothing over the
per-element context MLP - consistent with section 0.4's skepticism about spatial
structure. Loss plateaus suggest a genuine conditional-entropy ceiling near ~62-63%
FOR THIS context family (row-segment + group/row/col statistics), i.e. most of QAT's
deviation from the magnitude rule looks unpredictable from base-local information.
The 0.73-hard-acc runs show the model DOES know where deviations are likely - it
just can't beat the base rate pointwise without sacrificing easy entries.

### What this means for the paper, and next steps in priority order

1. The section-9 "pessimistic case" framing is now the working hypothesis:
   "ternary QAT's code is ~60% magnitude-rule + a large, locally-unpredictable
   remainder" - still workshop-grade IF backed by (a) the noise-floor measurement
   (section 6.4's own multi-seed QAT at 0.6B - now the single most important
   experiment: if two same-recipe QAT runs also agree only ~62% beyond the rule,
   the map is as predictable as it can possibly be and the framing flips to
   "QAT outcome is noise beyond the rule"), and (b) richer context ablations
   (whole-row/whole-column context, cross-layer features, larger models) to
   defend the "not predictable" claim against "your context was too small".
2. Functional eval (PoC step 4: assemble.py + perplexity/KL) remains worth building:
   the learned SCALES are clearly better (relerr 0.128 vs 0.193), and scales may
   matter more functionally than the marginal code gains; the oracle decompositions
   (section 4.2) will show this cheaply.
3. The binary-family pair and the 4B/8B scale ladder are unblocked (same tools, new
   --qat_model_id / --expected_family) and cheap to extract; cross-scale consistency
   of these numbers strengthens whichever framing survives.

### 10b. PoC step-4 results (added 2026-07-19)

**Framework overfit checks (tools/overfit_single_tensor.py)**: position-embedding
memorization of a real tensor reaches **1.0000** (PASS - training machinery
verified); the SAME setup with only the position-blind context features - trained
and evaluated on the same single tensor - caps at **0.6713**. Together: the ~62%
plateau is an information ceiling of base-local features, not a capacity/optimizer
problem. LR sweep confirms: lr 3e-4/1e-3/3e-3 all land at 0.622-0.624 (serials 3/5/6).

**Binary pair (Bonsai-1.7B-unpacked, serials 7/8)**: sign baseline 0.7160 overall -
QAT flips 28.4% of signs. Learned models: 0.7184/0.7189 = at/below the sign
baseline's own-split value (0.7190). Binary sign flips look essentially
unpredictable from local context - same story as ternary, cleaner setting.

**Functional perplexity (WikiText-2, 145x2048 tokens, shared Bonsai skeleton)**:

| blocks | PPL |
|---|---|
| fp (Qwen3 blocks) | 27.0 |
| oracle (real Bonsai blocks; sanity) | 17.7 |
| naive_b0 per-tensor absmean PTQ | 349,604 |
| naive_b1 per-group absmean PTQ | 879,504 |
| predicted (weight-map, serial 4) | 145,227 |

Readings: (1) the stated goal "surpass naive mean/max-abs quantization" is met -
the learned map is 2.4-6x lower PPL than the naive PTQ variants (1.8 nats better
nll than naive_b1); (2) but ALL calibration-free ternarizations of Qwen3 blocks are
catastrophically broken (PPL >1e5 vs oracle 17.7) - at ~62% code accuracy the model
is gibberish; the remaining 38% IS the model. This makes the ceiling question
(section 10, next step 1: multi-seed own-QAT noise floor) decisive for the paper,
and makes "weight-map as QAT init / partial-QAT accelerator" (section 6.3) the only
practically-plausible use of the map - pure zero-shot conversion is dead on
functional grounds, for ANY method capped near this accuracy, learned or not.

Ops note: the 2026-07-19 box reboot during stage lr_3e-3 was a kernel 'Bad page
state' fault in ext4 mmap readpage on a shard file - a kernel/hardware-level issue
(not an application error); if it recurs under heavy page-cache churn, suspect RAM.
