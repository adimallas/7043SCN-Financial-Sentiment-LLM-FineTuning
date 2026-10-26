# Task 1 Independent Results Audit

**Student:** Sampath Adimalla (17522555)  
**Module:** 7043SCN — Generative AI and Reinforcement Learning  
**Analysed evidence:** `Sampath_17522555_Task1_GitHubSafe_Results.zip` (55 files)

## Overall conclusion

**PASS — the numerical experiment outputs are internally consistent.** This means the research experiments are ready to inform the Task 1 paper. It does **not** mean the assignment is ready to submit: the full notebook/code, Coventry GitHub milestone history, accessible repositories, linked Task 2 project and two viva recordings remain separate submission requirements. This is an audit of the provided logs and prediction files, not an independent rerun of GPU training.

## Test results (518 examples; standard prompt)

| Model | Accuracy | Macro F1 | 95% bootstrap CI for Macro F1 | Training seconds | Trainable parameters | Recorded peak allocated GPU GiB |
|---|---:|---:|---|---:|---:|---:|
| Pre-trained baseline | 0.33784 | 0.33233 | [0.29942, 0.36421] | 0 | 0 | 0 (no training) |
| Full fine-tuning | 0.91892 | 0.91333 | [0.88156, 0.93979] | 360.30 | 76,961,152 | 1.7468 |
| LoRA r=16 | 0.91699 | 0.90851 | [0.87660, 0.93547] | 192.87 | 688,128 | 0.7566 |

Source: `reports/final_test_metrics.csv`, `macro_f1_bootstrap_ci.csv`.

The full FT–LoRA Macro F1 difference is only 0.00482 in favour of full FT. The paired bootstrap difference (LoRA minus full FT) is -0.00501, 95% interval [-0.02141, +0.00945]. **The interval includes zero**, so there is no strong evidence that the small observed performance difference is reliable at the 95% confidence level under this resampling procedure. LoRA used **99.11% fewer trainable parameters**, **46.47% less recorded training time**, and **56.69% less recorded peak GPU memory allocation**; peak allocated VRAM is not total system VRAM or a guaranteed saving across devices.

## Dataset, selection and integrity checks

- Financial PhraseBank `Sentences_75Agree`: 3,453 raw rows, 5 duplicates removed, **3,448 unique rows**.
- Stratified splits: train **2,412**, validation **518**, test **518**. There are **no duplicate `row_id` values across partitions**.
- Validation Macro F1: LoRA r8 **0.88990**; LoRA r16 **0.89429**. Thus r16 selection is consistent with the saved validation-only selection protocol.
- Test predictions align by identical 518 row IDs; independently recomputed accuracy, class F1, macro F1, and error-count matrices agree with the report files for all three evaluated models.
- Full FT made **42** errors; LoRA r16 made **43** errors. Their predictions are identical on **509/518** test cases.
- Training logs reach **755 steps** in each trained run; their lowest recorded validation losses match the training JSON metadata.
- ZIP integrity check (`ZipFile.testzip`) **passed**.

## Essential critical findings for the paper

1. **Domain adaptation is decisive:** the untuned baseline never predicted neutral on the held-out test data, where neutral is **322/518 (62.16%)** of examples. Its 33.78% accuracy is even lower than a trivial majority-label predictor (62.16% accuracy). The Macro F1 comparison remains useful, but the majority-label accuracy caveat must be stated.
2. **LoRA approaches full FT:** test Macro F1 was 90.85% versus 91.33%, while the trainable parameter counts differ dramatically. Do not call LoRA statistically worse on this sample.
3. **Prompt sensitivity is substantial:** under the alternative held-out prompt, Macro F1 fell to **0.80875** for full FT and **0.63567** for LoRA r16 (baseline: **0.23189**). Discuss prompt dependence candidly rather than claiming robustness.
4. **Only one random seed:** 1,000 bootstrap resamples capture uncertainty in this test sample, not variability between independently trained models. Avoid claims about run-to-run reproducibility without repeated training.
5. **A possible hardware-precision reporting caveat:** saved logs show a **Tesla T4** with `bf16=True`. T4 lacks native BF16 tensor-core support; modern PyTorch's `is_bf16_supported()` can return true when BF16 is **emulated**. Record this accurately and do not claim native BF16 acceleration without additional checks. See: https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html .
6. **GitHubSafe ZIP scope:** raw training code, pretrained/fine-tuned weights and runtime checkpoint directories are not in this archive. Keep the original notebook and separately backed-up weights; do not claim this ZIP alone provides end-to-end reproduction.
7. **Reference and licence:** attribute Malo et al. (2014), DOI 10.1002/asi.23062; Financial PhraseBank carries CC BY-NC-SA 3.0 conditions. Avoid distributing the raw licensed text in the repository unless licence compliance has been reviewed.

## Final QA status

- **Passed:** archive integrity, dataset split consistency, actual metrics reconstruction, error report reconciliation, LoRA rank selection logic, training log metadata and prompt sensitivity results.
- **Remaining:** verification of final public/private GitHub URL access, source notebook checked into the right university repo, complete paper with traceable tables/figures and cited literature, both viva videos, Task 2 repository link, and final Aula upload.

All numeric statements above are grounded in the uploaded ZIP and independently recalculated from its predictions. This is an output audit, not a second GPU training run or a predicted assessment grade.
