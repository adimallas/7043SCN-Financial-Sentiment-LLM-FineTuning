# Model card — 7043SCN Task 1

**Student:** Sampath Adimalla (17522555)  
**Application:** English financial news sentiment classification  
**Base model:** [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) — encoder–decoder text-to-text transformer  
**Data source:** Financial PhraseBank (Malo et al., 2014), Sentences_75Agree, downloaded from [https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip](https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip)  
**Dataset licence:** CC BY-NC-SA 3.0; maintain attribution and non-commercial licence conditions.  
**Classes:** negative / neutral / positive  
**Data partitions:** train=2412, validation=518, test=518.  
**Selection:** LoRA rank selected using validation macro-F1; final test was reserved for one post-selection evaluation.  
**LoRA rank selected:** lora_r16.  
**Hardware:** Tesla T4 (14.56 GiB).  
**Seed:** 42.  

## Methods
Pretrained zero-update baseline; all-parameter supervised fine-tuning; LoRA with q/v target modules at ranks [8, 16]. Training checkpoint selection used validation loss. See runtime_config.json, training reports and learning curves.

## Quantitative evaluation
Actual measured results: `reports/final_test_metrics.csv`; bootstrap intervals: `reports/macro_f1_bootstrap_ci.csv`; paired differences: `reports/paired_bootstrap_differences.csv`.

## Known limitations
* Data represent short financial news sentences; not contemporary markets or investment outcomes.
* Sentence-level stratified random partition can share vocabulary or economic entities; generalisation over time is not evaluated.
* Annotation disagreement outside the chosen subset is excluded from the principal evaluation.
* Three sentiment labels are not a calibrated financial signal or investment recommendation.
* A single initialisation seed does not measure full between-run training variability.
* Prompt sensitivity tests alternate instructions on the same held-out news, not out-of-distribution companies or news events.
* Invalid generated labels count as errors and are reported separately.

## Social, ethical, legal, and professional use
Bias from historical financial coverage, sector imbalance and investor-perspective labelling limits applicability. Human oversight is required; do not use for automatic trading or regulated financial decisions. Respect the source licence, attribution, disclosure of research limitations and relevant data processing obligations.

## Reproducing
Execute the accompanying Colab notebook on an NVIDIA GPU with its exact package versions. Do not fit or tune on the final test labels. See the notebook for command-level settings, data SHA256, random seed, implementation and saved results.
