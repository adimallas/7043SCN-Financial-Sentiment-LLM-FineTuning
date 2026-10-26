# Task 1 experimental protocol — Sampath Adimalla (17522555)

## Research question
How do full fine-tuning and LoRA compare to the untuned FLAN-T5 Small model on the accuracy, error characteristics and training-resource cost of financial-news sentiment classification?

## Hypotheses (to be tested, not presumed true)
- **H1:** Both adaptation approaches improve held-out macro F1 over the untuned model.
- **H2:** LoRA trains fewer model parameters and uses less peak allocated GPU memory than full fine-tuning, subject to measured values.
- **H3:** Increasing LoRA rank from 8 to 16 changes the validation macro F1 and training resource trade-off.

## Analysis protocol
1. Download the original published Financial PhraseBank ZIP, log its SHA256, use only the 75%-agreement subset for primary experiments.
2. Parse source labels, remove exact normalized duplicates, detect conflicting labels and create one seeded stratified train/validation/test split (70/15/15).
3. Evaluate zero-update baseline on validation, with exact label parsing. Record invalid generations as errors.
4. Full fine-tune all model parameters when GPU memory allows; train LoRA ranks 8 and 16 from the same original pre-trained weights. Select checkpoints by validation loss and LoRA rank by validation macro F1.
5. After model selection, evaluate baseline, full fine-tuning and selected LoRA on the locked test set once, using identical prompts, generation parameters and ordering.
6. Report accuracy, macro and class-specific F1, confusion matrices, resource time and peak allocated GPU memory. Calculate paired bootstrap uncertainty intervals over the same held-out examples.
7. Run a second instruction template on the same held-out test sentences as a limited prompt sensitivity diagnostic; do not call this an external or temporal generalisation test.
8. Discuss failures, limitations, model risks and dataset licence. Preserve original logs and outputs in a versioned archive.

## Fairness and limitations
- Epoch count, effective batch size, dataset and decoding are aligned. Learning rates differ as method-specific hyperparameters; comparisons are method/configuration combinations, not a strictly isolated causal effect of the adaptation algorithm.
- LoRA ablation uses validation performance for rank choice, rather than selecting on the test set.
- One seed for each run does not quantify training-seed variation. Bootstrap intervals quantify test-sample uncertainty only, not variation between independent training runs.
- Historical financial sentences are not a measure of contemporary investment returns or safe financial advice.
- Do not invent metrics before executing Colab; report only observed results.

## Required evidence for report and viva
`source_manifest.json`, `runtime_config.json`, `split_manifest.csv`, `validation_model_comparison.csv`, `adapter_selection.json`, `*_training.json`, `*_history.csv`, `final_test_metrics.csv`, `macro_f1_bootstrap_ci.csv`, `paired_bootstrap_differences.csv`, `prompt_sensitivity.csv`, confusion-matrix images, MODEL_CARD.md and training checkpoint metadata.

## Paper mapping
- **Data/method:** Source SHA256, split counts, architecture, training configuration, LoRA target modules and seed.
- **Results:** Held-out scores with uncertainty, per-class confusion patterns, efficiency and rank validation comparison.
- **Discussion:** Error taxonomy, invalid outputs, prompt sensitivity, resources and limitations.
- **Ethics/legal:** CC BY-NC-SA 3.0 attribution, domain bias, historical label limitation and human oversight.
