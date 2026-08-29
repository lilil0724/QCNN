# Session conventions: Qwen3 vs. Ternary-Bonsai-1.7B analysis

This repo is a focused spinoff from `QuantizedASR` (the larger ASR-quantization
project these tools were originally built in), scoped specifically to comparing
`Qwen/Qwen3-1.7B` (base, full-precision) against `prism-ml/Ternary-Bonsai-1.7B-unpacked`
(the same architecture, ternary-QAT-trained) - the cleanest same-architecture
ternary-QAT comparison available. See `docs/BONSAI_QWEN3_1.7B_FINDINGS.md` for prior
results and findings, including an important checkpoint-naming gotcha (read it before
running anything against a new `prism-ml/*Bonsai*` checkpoint).

Nothing here has been copied *from* this repo back into `QuantizedASR` - it's a
one-way spinoff, and the two are meant to diverge freely as this repo's own
experiments grow. The conventions below are distilled from real problems hit while
building the source tooling in `QuantizedASR`; they're kept here because the
underlying lessons are general, not because this repo does what `QuantizedASR` does.

---

## 1. Verification discipline (the most important section)

**A truncated log is not evidence of success.** `tail -N` on a long-running command
can silently miss a real failure buried earlier in the output - read the whole thing,
or at minimum grep for error/traceback markers across the full log, not just the end.

**Before recommending or reporting anything recalled from memory, a summary, or an
older doc, re-verify it against the current, real state** (grep for the file, check
the function still exists, re-run the comparison). A memory or doc saying "X exists"
or "this was already measured" is a claim about the state of the world *when it was
written* - it can go stale, especially across a checkpoint update or a re-run with
different arguments.

**Real GPU verification beats reasoning about code.** When in doubt whether a
comparison script produces the right numbers, run it for real against the actual
checkpoints and inspect the actual output - don't infer correctness from reading the
code or from a prior claim.

---

## 2. Failure triage discipline

**Don't guess the cause - read the actual traceback and check the actual system
state.** A "the script crashed" symptom can have many genuinely different root causes
(an HTTP timeout resolving a checkpoint, a real host-RAM/GPU-memory OOM, a shape/dtype
mismatch between two model versions, a wrong-checkpoint mixup - see the erratum in
`docs/BONSAI_QWEN3_1.7B_FINDINGS.md` for a real example of the last one). Each needs
its own piece of confirming evidence (the actual stack trace, `nvidia-smi`, `dmesg`,
directly checking which exact HF Hub repo was loaded) - don't pattern-match against a
previous, superficially similar failure.

**Reproduce before concluding "transient" or "deterministic."** One failure is not
enough evidence either way.

---

## 3. Retry and exclusion policy

- On a transient-looking failure, retry once before concluding anything.
- If a fix is found (e.g. a smaller batch size resolves a real OOM), apply and verify
  the fix before working around it - a workaround is the fallback when nothing fixes
  the underlying issue, not the first move.
- If a checkpoint, layer, or comparison has to be skipped/excluded for a real reason,
  say so explicitly in the output/report (with the reason) rather than silently
  dropping it - a silent gap reads as "fully covered" when it wasn't.

---

## 4. Multi-environment editing hygiene

This repo currently lives **remote-only**, on the GPU box at
`/media/samsung/projects/qwen3-ternary-bonsai-analysis` - no local mirror is being
kept in sync right now. If that changes (a local mirror gets added later), the
following applies and matters a lot:

- **Every edit must be synced to every copy that matters** - never let them drift out
  of sync, even for a "quick" fix. Verify sync with a checksum (e.g. `md5sum`), not
  just an assertion that a copy command succeeded.
- **Syntax-check every script after editing** (`python -m py_compile <file>`),
  especially after a scripted/patch-based edit rather than a direct manual one.
- **When patch logic needs to run over SSH, write it to a local file and `scp` it over
  rather than embedding it inline in a double-quoted SSH command string** - backticks
  and `$(...)` inside a double-quoted argument get expanded by the *local* shell
  before SSH ever sees them, silently corrupting the intended remote content.
- **After any multi-step automated patch, re-read the actual resulting file content**
  - don't trust the patch script's own "success"/"updated" print as proof the content
  is correct.

---

## 5. Change hygiene

- **Prefer additive new files over rewriting existing ones**, and ask before touching
  a pre-existing file that wasn't already created/touched earlier in the same
  session.
- **Never take destructive or hard-to-reverse actions without explicit permission.**
- **Never download anything to the home directory** - use an explicit, designated
  storage path on every command that touches an HF Hub model/dataset cache, since
  environment variables don't necessarily persist across non-interactive shells.

---

## 6. A concrete technical gotcha worth remembering

A resource-ceiling failure (host RAM or GPU memory) does not always present as a
clean, recognizable "out of memory" error - it can look like a segfault, a hang, or an
unrelated-looking downstream error. When a run against a large checkpoint (1.7B+
parameters, two of them loaded at once for a comparison) fails in a confusing way,
check actual resource usage directly (`nvidia-smi`, `dmesg -T | tail`) rather than
pattern-matching the symptom.

---

## 7. HF cache and data storage on this server

- **All HF Hub downloads (models, datasets) go to the designated cache at
  `/home/pcs5060ti/Desktop/hf`** - set `HF_HOME=/home/pcs5060ti/Desktop/hf` and
  `HF_HUB_CACHE=/home/pcs5060ti/Desktop/hf/hub` explicitly on every command that touches
  the Hub (env vars don't persist across non-interactive shells; this repo's tools
  under `tools/` also set these as defaults themselves when unset). Never let
  anything fall through to another cache location, and don't put large artifacts on
  `/media/samsung` either (it is nearly full) - `/hdd` is the big disk.
- Derived data (extracted pair shards etc.) lives under `/hdd/edwin/qwen3vsbonsai/`
  (e.g. `pairs/<pair slug>/` from `tools/extract_pair.py`).
