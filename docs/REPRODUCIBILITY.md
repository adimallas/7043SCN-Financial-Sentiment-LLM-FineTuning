# Computational reproducibility and provenance

The code and results are separated to distinguish a recorded experiment from a fresh rerun. Reproducing the *exact* reported numbers may depend on the underlying Colab hardware, CUDA runtime, package builds and deterministic kernel implementation.

## Original run
- Model: `google/flan-t5-small`, encoder-decoder, 76,961,152 base parameters.
- Data: Financial PhraseBank, 75%-agreement, 3,448 deduplicated records.
- Split: 2,412 train; 518 validation; 518 test (seed 42).
- Training: five epochs; batch 8; accumulation 2; full FT learning rate `5e-5`; LoRA learning rate `3e-4`; LoRA ranks 8, 16; alpha 16; dropout 0.05.
- Environment: NVIDIA Tesla T4, ~14.56 GiB GPU memory; exact library and CUDA versions recorded in `results/artifacts/runtime_config.json`.
- Model selection: best checkpoint monitored on validation loss; LoRA rank selected by validation macro F1 before evaluating held-out test once.

## Recorded evidence files
- `results/artifacts/source_manifest.json`: archive origin and SHA256 hash.
- `results/artifacts/split_manifest.csv`: anonymised hashes, labels and split membership.
- `results/reports/validation_model_comparison.csv`: all validation comparisons, including rank 8.
- `results/reports/final_test_metrics.csv`: locked test results.
- `results/reports/paired_bootstrap_differences.csv`: paired uncertainty intervals.
- `results/predictions/`: actual per-example reference and predicted labels.
- `results/logs/`: saved training histories.

`python scripts/verify_results.py` independently recomputes accuracy and macro F1 from those predictions. It does not re-train a model or validate future deployment performance.

## Limitations of experimental control
- One seed is insufficient for estimating variation between independent training runs.
- The input data are historical, short and relatively unambiguous; repeated financial entities and lexical features can be shared across a random split.
- Method-specific learning rates differ; the comparison reflects selected configurations rather than a perfectly isolated fine-tuning-algorithm treatment effect.
- The recorded peak allocated GPU memory is not total process memory or peak power draw.
- Two prompts are a limited sensitivity test; they do not estimate robustness to all possible phrasing.
- The original ZIP excludes trained weights to keep size manageable. Reproduction from pre-trained weights requires a new GPU run or separately backed up original checkpoints.
