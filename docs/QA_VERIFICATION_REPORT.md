# Task 1 pre-run verification record (24 October 2026)

**Student:** Sampath Adimalla (17522555)
**Module:** 7043SCN, Task 1

## Checks performed
- Inspected every code cell and the matching standalone Python script.
- Verified notebook JSON schema and parsed all 15 Python code cells (excluding the Colab pip magic) without syntax errors.
- Confirmed exact published model / dataset repository file paths from their official Hugging Face pages.
- Confirmed pinned Transformers / PEFT / Datasets versions exist on PyPI and that Seq2SeqTrainingArguments exposes `eval_strategy` and `predict_with_generate`.
- Executed 12 local unittest checks: 5 structural integrity checks and 7 offline scientific smoke checks. All passed.
- Exercised prompt parsing, label scoring including invalid generations, deterministic stratified splitting, and paired bootstrap arithmetic with synthetic data.
- Examined checkpoint save/reload logic, adapter selection, confusion matrices, error outputs and portable ZIP construction.

## Corrections made
- Fixed missing `display()` support in the standalone Python entrypoint.
- Added a package-version guard so Colab reports stale dependencies clearly after installation.
- Added a robust official Hugging Face Hub downloader with direct-URL fallback, ZIP validity / CRC checks.
- Added an actual two-example forward-pass / finite-loss preflight before lengthy training.
- Explicitly require published Safetensors weights.
- Validate the existence of full-model/LoRA adapter files after training.
- Added a final completeness audit verifying the expected metric, prediction, figure and log files.

## What cannot be certified without Google Colab
- We cannot install the pinned ML stack or download the model/dataset from this execution environment due to network restrictions.
- We do not have access to an NVIDIA GPU in this environment for actual forward/backward training with Hugging Face packages.
- We cannot certify specific accuracy, F1, training runtime, no GPU OOM, Colab availability or external CDN reliability.
- A clean final GPU run is still mandatory before interpreting any result or writing a paper based on the outputs.

## Go/no-go signals in Colab
1. Dependencies imported at the pinned versions; GPU name and memory printed.
2. Dataset archive verified; nonzero class counts and no duplicate split rows.
3. Finite two-example preflight loss printed.
4. FFT and two LoRA variants complete, and validation table appears.
5. Final held-out metrics, bootstrap intervals, figures and error reports generated.
6. `FINAL AUDIT PASSED` printed and downloadable results ZIP created.

**Assessment:** Local static and synthetic checks passed; live GPU execution remains UNVERIFIED.
