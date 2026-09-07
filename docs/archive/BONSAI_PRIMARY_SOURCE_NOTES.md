# Ternary Bonsai production details: primary-source audit

Research date: 2026-08-21

## Scope and evidentiary rule

This note asks a narrow question: what can be stated, from first-party sources, about how
`prism-ml/Ternary-Bonsai-1.7B-unpacked` was produced from Qwen3-1.7B? It uses Prism ML's
model cards, announcement, and April 2026 white paper, plus official Qwen/Hugging Face
configuration and implementation sources. It does **not** treat community posts or this
repository's reverse-engineering observations as evidence of Prism ML's training recipe.

The main conclusion is important for a paper: **Prism ML discloses the deployed weight
representation and its coverage, but does not disclose a reproducible training/QAT or
distillation recipe.** Consequently, claims about datasets, training tokens, teacher models,
losses, optimizer, schedule, or straight-through estimators must be marked "not disclosed,"
not filled in from generic BitNet practice.

## What Prism ML explicitly discloses

### Starting model and architecture

- The Ternary Bonsai family is built from the dense decoder-only Qwen3 family; the 1.7B
  member is built from Qwen3-1.7B. Prism ML says the underlying architectures are unchanged
  and that the novelty lies in the weight representation. See the official
  [Ternary Bonsai white paper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf),
  pp. 2 and 4.
- The official Qwen3-1.7B configuration specifies `Qwen3ForCausalLM`, 28 decoder layers,
  hidden size 2048, intermediate size 6144, 16 query heads, 8 KV heads, head dimension 128,
  tied input/output embeddings, and vocabulary size 151,936. See the
  [Qwen3-1.7B config](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json).
- The unpacked Bonsai repository describes itself only as an FP16-safetensors expansion of
  the ternary model for stock Hugging Face tooling. It does not expose continuous pre-QAT
  master weights or a training checkpoint. See the
  [unpacked model card](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-unpacked).

### Ternary representation and scales

Prism ML gives the reconstruction equation

\[
  w_i = s_g t_i, \qquad t_i \in \{-1,0,+1\},
\]

where one FP16 scale \(s_g\) is shared by each group of 128 weights. The authors call this
"ternary g128." The code carries \(\log_2 3 \approx 1.585\) bits/weight, and the FP16 scale
amortizes to \(16/128=0.125\) bits/weight, for an idealized 1.71 bits/weight. See the
[white paper, Section 2.1](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)
and the first-party [announcement](https://prismml.com/news/ternary-bonsai).

The GGUF release uses `Q2_0` packing. Each 128-weight block occupies 34 bytes: 32 bytes for
128 two-bit codes and 2 bytes for an FP16 scale, or 2.125 physical bits/weight. Codes are
decoded as `(q - 1) * scale`; the fourth two-bit code is unused by ternary weights. The GGUF
card calls this conversion lossless with respect to the already-ternary FP16 checkpoint. See
the official [Ternary-Bonsai-1.7B GGUF card](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf).

Two distinctions matter:

1. 1.585 bits is ternary information content, 1.71 bits includes ideal FP16 scale overhead,
   and 2.125 bits is the physical `Q2_0` packing cost. These numbers answer different questions.
2. The first-party sources specify group size 128, but do **not** specify the exact flattening
   order or whether groups are formally defined as row-contiguous 128-column chunks. Any such
   orientation claim must be attributed to checkpoint inspection, not to the white paper.

### Which components are ternary

Prism ML states that ternary weights are used in:

- token embeddings;
- attention projections;
- MLP projections; and
- the language-model head.

It further states that normalization parameters and scale metadata remain at higher precision.
This is the precise interpretation of the marketing phrase "throughout the entire network":
all major matrix-heavy weights are ternary, while small normalization vectors and scale
metadata are not. Sources: [white paper, Table 1 and Section 2.1](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)
and [Prism ML announcement](https://prismml.com/news/ternary-bonsai).

For Qwen3, "attention projections" maps to `q_proj`, `k_proj`, `v_proj`, and `o_proj`, while
"MLP projections" maps to `gate_proj`, `up_proj`, and `down_proj`. This naming expansion is
an inference from the official
[Hugging Face Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py),
not a tensor-by-tensor list published by Prism ML. With 28 blocks, the transformer body has
28 instances of each of those seven projection matrices. The input embedding and LM head are
tied in the original Qwen3 configuration; Prism ML does not explain how tying is represented
during ternary training or export.

The following are therefore supported for a Methods section:

| Component | Primary-source status |
|---|---|
| `model.embed_tokens.weight` | Included by Prism ML's "embeddings" statement |
| `self_attn.{q,k,v,o}_proj.weight`, all 28 blocks | Included; exact names inferred from Qwen3 implementation |
| `mlp.{gate,up,down}_proj.weight`, all 28 blocks | Included; exact names inferred from Qwen3 implementation |
| `lm_head.weight` | Included by Prism ML's explicit statement |
| RMSNorm weights, including block norms and Q/K norms | Higher precision; not ternary |
| Per-group scale metadata | FP16 |
| Biases | Qwen3-1.7B projection layers are configured without attention bias and the standard MLP projections are bias-free; Prism ML gives no separate bias policy |

## What is not disclosed

The April 2026 announcement says that full technical training details are in the white paper,
but the 11-page white paper contains no training-method section. It covers representation,
storage, deployment throughput/energy, and evaluation. Neither it nor the 1.7B model cards
provide the following:

Prism ML's earlier
[1-bit Bonsai white paper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf),
Section 3 (p. 5), is more explicit about the broader Bonsai method: it says the foundation
comes from proprietary Caltech intellectual property. That statement supports describing the
method family as proprietary, but it still does not reveal the algorithm, and it is not proof
that every training detail was identical for the later ternary release.

| Reproducibility item | Status in Prism ML primary sources |
|---|---|
| PTQ versus QAT versus continued pretraining under quantization | **Not disclosed** |
| Continuous latent/master-weight parameterization | **Not disclosed** |
| Straight-through estimator or alternative gradient estimator | **Not disclosed** |
| Formula used to choose ternary codes during training | **Not disclosed** |
| Formula used to estimate/update the g128 scale during training | **Not disclosed** |
| Whether activations were quantized during training | **Not disclosed** |
| Training stages or progressive bit-width schedule | **Not disclosed** |
| Dataset names, composition, filtering, or mixture weights | **Not disclosed** |
| Number of training examples or tokens | **Not disclosed** |
| Sequence length used for training | **Not disclosed** |
| Number of steps/epochs and global batch size | **Not disclosed** |
| Optimizer, learning rate, scheduler, weight decay, clipping | **Not disclosed** |
| Hardware, wall-clock time, or training compute | **Not disclosed** |
| Knowledge-distillation teacher | **Not disclosed** |
| Token-level cross-entropy, logit-KD, feature matching, or other objective | **Not disclosed** |
| Loss coefficients or temperatures | **Not disclosed** |
| Random seeds and run-to-run variance | **Not disclosed** |
| Layerwise exceptions beyond high-precision norms/metadata | **Not disclosed** |

Therefore, a paper may accurately say that the public checkpoint is *a ternary model derived
from Qwen3-1.7B with g128 FP16-scaled weights across the major matrix components*. It should
not say that Prism ML used a specific QAT, distillation, or dataset recipe unless Prism ML
publishes additional evidence. In particular, this repository's phrase "ternary-QAT-trained"
is a working assumption or reverse-engineering interpretation, not a fact established by the
currently available primary sources.

## Audit of `tiiuae/onebitllms`

`onebitllms` is a first-party TII/Falcon-LLM toolkit for BitNet-style 1.58-bit training and
fine-tuning. It is useful as a comparator, but **it is not evidence of the Bonsai recipe**:
the Prism ML white paper, announcement, and 1.7B model cards do not identify it as the code
used to produce Ternary Bonsai.

The public toolkit does the following:

- `BitNetLinear.forward` quantizes activations and weights in the forward pass and uses the
  detach pattern `latent + (quantized - latent).detach()`, which is a straight-through
  estimator. See
  [`layers/bitnet.py`](https://github.com/tiiuae/onebitllms/blob/main/src/onebitllms/layers/bitnet.py).
- Its weight kernel computes one absolute mean over the entire flattened 2-D tensor, then
  applies `clamp(round(w / mean_abs), -1, 1) * mean_abs`. It is not a g128 weight rule. See
  [`kernels/weight_quant.py`](https://github.com/tiiuae/onebitllms/blob/main/src/onebitllms/kernels/weight_quant.py).
- Its activation kernel uses per-row/token absolute-max scaling to signed int8, another detail
  that Prism ML does not disclose for Bonsai. See
  [`kernels/activation_quant.py`](https://github.com/tiiuae/onebitllms/blob/main/src/onebitllms/kernels/activation_quant.py).
- Its recursive training-time replacement changes `nn.Linear` modules but explicitly skips
  `lm_head`; embeddings are not `nn.Linear` and are therefore not replaced. See
  [`utils/monkey_patching.py`](https://github.com/tiiuae/onebitllms/blob/main/src/onebitllms/utils/monkey_patching.py).
- Its final checkpoint utility explicitly excludes `model.embed_tokens.weight`, the final
  model norm, `lm_head.weight`, input layer norms, and post-attention layer norms. See
  [`utils/quantization_utils.py`](https://github.com/tiiuae/onebitllms/blob/main/src/onebitllms/utils/quantization_utils.py).
- The README's concrete SFT example concerns Falcon-E plus the Capybara dataset and is not a
  Bonsai training run. See the official
  [`onebitllms` README](https://github.com/tiiuae/onebitllms).

These differences are substantive: Ternary Bonsai claims g128 scales and ternary embeddings
and LM head, whereas the public `onebitllms` path uses a tensor-wide weight scale and leaves
embeddings and LM head unquantized. A paper must not use `onebitllms` defaults to fill gaps in
Prism ML's undocumented recipe.

## Recommended wording for a paper

> We study Prism ML's public `Ternary-Bonsai-1.7B-unpacked` checkpoint, which is derived from
> Qwen3-1.7B while retaining the Qwen3 decoder architecture. According to Prism ML, its major
> matrix weights—including token embeddings, all attention and MLP projections, and the
> language-model head—are represented by ternary codes in \(\{-1,0,+1\}\), with one FP16
> scale shared per group of 128 weights; normalization parameters remain in higher precision.
> Prism ML does not publish the training data, token count, optimization schedule, gradient
> estimator, distillation teacher, or training objectives. We therefore treat the checkpoint
> as an observed endpoint and do not assume a particular QAT or distillation procedure.

If the manuscript needs a fully reproducible account of Bonsai's production, the missing
information must come from the model authors. The minimum useful author query is: (1) PTQ or
QAT/continued training; (2) the exact forward quantizer and STE; (3) group ordering and scale
estimator; (4) trainable versus frozen tensors; (5) data mixture and token count; (6) teacher
and KD losses; (7) optimizer/schedule/batch/steps; and (8) seeds and compute.

## Primary sources

1. Prism ML, [Ternary Bonsai 8B white paper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf), 2026-04-16.
2. Prism ML, [1-bit Bonsai 8B white paper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf), 2026.
3. Prism ML, [Introducing Ternary Bonsai](https://prismml.com/news/ternary-bonsai), 2026-04-16.
4. Prism ML, [`Ternary-Bonsai-1.7B-unpacked` model card](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-unpacked).
5. Prism ML, [`Ternary-Bonsai-1.7B-gguf` model card](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf).
6. Qwen, [`Qwen3-1.7B` configuration](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json).
7. Hugging Face Transformers, [Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py).
8. TII Falcon-LLM Team, [`onebitllms`](https://github.com/tiiuae/onebitllms).
