"""Equivalent CLI pipeline for the Google Colab research notebook.
Run with python task1_pipeline.py after installing requirements in a GPU environment.
"""
import os, sys, re, json, math, gc, time, random, hashlib, zipfile, io, platform, inspect, shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import sklearn
import torch
import transformers, datasets, peft, accelerate, sentencepiece
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj.to_string(index=False) if hasattr(obj, 'to_string') else str(obj))

# If Colab preloaded another version before the installation cell, restart its runtime.
_EXPECTED = {'transformers':'4.57.1','datasets':'4.4.1','peft':'0.18.0','accelerate':'1.11.0'}
for _package, _version in _EXPECTED.items():
    _actual = __import__(_package).__version__
    if _actual != _version:
        raise RuntimeError(f'{_package} {_actual} is loaded, expected {_version}. Restart Colab runtime and run all cells again.')
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, f1_score)
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, set_seed)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

STUDENT = {'name': 'Sampath Adimalla', 'student_id': '17522555', 'module': '7043SCN'}
MODEL_ID = 'google/flan-t5-small'
SEED = 42
CONFIG = {
    'agreement': '75',
    'test_fraction': 0.15,
    'validation_fraction': 0.15,
    'input_max_tokens': 128,
    'target_max_tokens': 8,
    'generation_max_new_tokens': 6,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 2,
    'eval_batch_size': 16,
    'epochs': 5,
    'learning_rate_full': 5e-5,
    'learning_rate_lora': 3e-4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'lora_ranks': [8, 16],
    'bootstrap_samples': 1000,
    'run_full_finetuning': True,
}
assert 0 < CONFIG['validation_fraction'] < 1 and 0 < CONFIG['test_fraction'] < 1
assert CONFIG['validation_fraction'] + CONFIG['test_fraction'] < 1
assert CONFIG['lora_ranks'] and len(CONFIG['lora_ranks']) == len(set(CONFIG['lora_ranks']))
OUT = Path('/content/Sampath_17522555_Task1') if Path('/content').exists() else Path.cwd() / 'Sampath_17522555_Task1'
for sub in ['data_cache', 'artifacts', 'models', 'reports', 'figures', 'predictions', 'logs']:
    (OUT/sub).mkdir(parents=True, exist_ok=True)
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE != 'cuda':
    raise RuntimeError('For final full fine-tuning please select a GPU in Colab: Runtime > Change runtime type > T4 GPU.')
GPU_NAME = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
BF16 = bool(torch.cuda.is_bf16_supported())
print('Device:', GPU_NAME, f'({GPU_GB:.2f} GiB)', 'bf16:', BF16)
print('Torch / CUDA:', torch.__version__, torch.version.cuda)
print('Transformers / PEFT:', transformers.__version__, peft.__version__)
ENV = {
    'utc_timestamp': datetime.now(timezone.utc).isoformat(),
    'student': STUDENT,
    'gpu': GPU_NAME,
    'gpu_memory_gib': round(GPU_GB,2),
    'cuda_runtime': torch.version.cuda,
    'python': sys.version,
    'versions': {'torch':torch.__version__, 'transformers':transformers.__version__,
                 'datasets':datasets.__version__, 'peft':peft.__version__,
                 'accelerate':accelerate.__version__, 'sklearn':sklearn.__version__,
                 'pandas':pd.__version__, 'numpy':np.__version__},
    'seed':SEED, 'model':MODEL_ID, 'configuration':CONFIG,
}
(OUT/'artifacts'/'runtime_config.json').write_text(json.dumps(ENV, indent=2, default=str))

SOURCE_URL = 'https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip'
ZIP_PATH = OUT/'data_cache'/'FinancialPhraseBank-v1.0.zip'
# Use Hugging Face's cached, verified transfer first; fall back to the direct public URL.
# Never silently reuse an incomplete download from an interrupted Colab session.
if ZIP_PATH.exists() and not zipfile.is_zipfile(ZIP_PATH):
    print('Existing dataset archive is invalid; downloading a fresh copy.')
    ZIP_PATH.unlink()
if not ZIP_PATH.exists():
    download_problems = []
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id='takala/financial_phrasebank',
                                 repo_type='dataset', filename='data/FinancialPhraseBank-v1.0.zip')
        shutil.copyfile(cached, ZIP_PATH)
    except Exception as exc:
        download_problems.append(f'HF Hub: {type(exc).__name__}: {exc}')
        try:
            response = requests.get(SOURCE_URL, timeout=120)
            response.raise_for_status()
            ZIP_PATH.write_bytes(response.content)
        except Exception as second_exc:
            download_problems.append(f'Direct URL: {type(second_exc).__name__}: {second_exc}')
    if not ZIP_PATH.exists() or not zipfile.is_zipfile(ZIP_PATH):
        ZIP_PATH.unlink(missing_ok=True)
        raise RuntimeError('Dataset download failed or returned an invalid ZIP. Check Colab Internet access. ' +
                           ' | '.join(download_problems))
with zipfile.ZipFile(ZIP_PATH) as _archive:
    corrupted_member = _archive.testzip()
    if corrupted_member is not None:
        raise RuntimeError(f'Corrupted dataset archive member: {corrupted_member}; delete {ZIP_PATH} and retry.')
source_sha256 = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
with zipfile.ZipFile(ZIP_PATH) as archive:
    choices = [n for n in archive.namelist() if n.lower().endswith(f"sentences_{CONFIG['agreement']}agree.txt".lower())]
    assert len(choices)==1, f'Unexpected archive layout: {choices}'
    original_text = archive.read(choices[0]).decode('iso-8859-1')

valid_labels = ('negative', 'neutral', 'positive')
rows=[]
for line in original_text.splitlines():
    if not line.strip():
        continue
    if '@' not in line:
        raise ValueError(f'Malformed line: {line[:80]}')
    sentence,label = line.rsplit('@',1)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    label = label.strip().lower()
    assert sentence and label in valid_labels, (sentence,label)
    rows.append({'sentence': sentence, 'label': label})
raw_df = pd.DataFrame(rows)
raw_count = len(raw_df)
raw_df['dedup_key'] = raw_df.sentence.str.casefold().str.strip()
conflict_counts=raw_df.groupby('dedup_key').label.nunique()
assert conflict_counts.max()==1, f'Conflicting labels for {(conflict_counts>1).sum()} normalized texts'
df=raw_df.drop_duplicates('dedup_key').reset_index(drop=True).copy()
df['row_id'] = df.dedup_key.map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()[:20])
assert df.row_id.is_unique
assert all(df.label.value_counts()>4), df.label.value_counts()

SOURCE = {'source_url': SOURCE_URL, 'archive_sha256':source_sha256,
          'archive_member': choices[0], 'raw_rows':raw_count, 'deduplicated_rows':len(df),
          'duplicate_rows_removed': raw_count-len(df), 'license':'CC BY-NC-SA 3.0',
          'reference':'Malo et al. (2014), Good debt or bad debt: Detecting semantic orientations in economic texts'}
(OUT/'artifacts'/'source_manifest.json').write_text(json.dumps(SOURCE,indent=2))
print('Raw:',raw_count,'Unique:',len(df),'Removed:',raw_count-len(df))
print('Class distribution:',df.label.value_counts().to_dict())
print('Source SHA256:',source_sha256)
display(df[['sentence','label']].head())

train_val_df, test_df = train_test_split(df, test_size=CONFIG['test_fraction'],
                                         random_state=SEED, stratify=df.label)
val_share = CONFIG['validation_fraction']/(1-CONFIG['test_fraction'])
train_df, val_df = train_test_split(train_val_df, test_size=val_share,
                                    random_state=SEED, stratify=train_val_df.label)
train_df, val_df, test_df = [x.reset_index(drop=True) for x in (train_df,val_df,test_df)]
sets = [set(x.row_id) for x in (train_df,val_df,test_df)]
assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]), 'Data leakage'
assert sum(map(len,(train_df,val_df,test_df)))==len(df)
for name,part in [('train',train_df),('validation',val_df),('test',test_df)]:
    print(name,len(part),part.label.value_counts().to_dict())
# No licensed sentence text is exported here. IDs are SHA-derived from canonical text.
manifest = pd.concat([part[['row_id','label']].assign(split=name) for name,part in
                      [('train',train_df),('validation',val_df),('test',test_df)]],ignore_index=True)
manifest.to_csv(OUT/'artifacts'/'split_manifest.csv',index=False)
class_counts = pd.crosstab(manifest['split'],manifest['label'])
class_counts.to_csv(OUT/'artifacts'/'split_class_counts.csv')
ax = class_counts.loc[['train','validation','test']].plot(kind='bar',figsize=(8,4),rot=0)
ax.set_title('Financial PhraseBank class distribution by data split')
ax.set_ylabel('Number of samples'); ax.set_xlabel('Partition')
plt.tight_layout(); plt.savefig(OUT/'figures'/'01_split_distribution.png',dpi=250); plt.show()

LABELS=['negative','neutral','positive']
LABEL2ID={name:i for i,name in enumerate(LABELS)}

def prompt(sentence, variant='standard'):
    if variant=='standard':
        return ("Classify the financial news sentence from an investor's perspective. "
                "Answer with exactly one word: negative, neutral, or positive.\n"
                f"Sentence: {sentence}\nSentiment:")
    if variant=='alternative':
        return ("Financial news sentiment task. Output only one of negative, neutral, positive.\n"
                f"News: {sentence}\nLabel:")
    raise ValueError('Unknown prompt variant')

def parse_label(generated):
    cleaned = re.sub(r'^[^a-z]+|[^a-z]+$', '', generated.strip().lower())
    return LABEL2ID.get(cleaned,-1)

assert [parse_label(x) for x in ('Positive.',' neutral ', 'NEGATIVE','unknown','positive neutral')] == [2,1,0,-1,-1]
TOKENIZER=AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
assert TOKENIZER.pad_token_id is not None

def generate_predictions(model, source_df, prompt_variant='standard'):
    model.eval()
    device = next(model.parameters()).device
    predictions=[]; generated=[]
    batch_size=CONFIG['eval_batch_size']
    for start in range(0,len(source_df),batch_size):
        batch=source_df.iloc[start:start+batch_size]
        input_strings=[prompt(s,prompt_variant) for s in batch.sentence]
        enc=TOKENIZER(input_strings,return_tensors='pt',padding=True,
                      truncation=True,max_length=CONFIG['input_max_tokens']).to(device)
        with torch.inference_mode():
            output=model.generate(**enc,do_sample=False,num_beams=1,
                                  max_new_tokens=CONFIG['generation_max_new_tokens'])
        texts=TOKENIZER.batch_decode(output,skip_special_tokens=True)
        generated.extend(texts)
        predictions.extend(parse_label(s) for s in texts)
    return predictions,generated

def summarise_predictions(source_df, preds, raw_outputs, model_tag, split_name, prompt_variant='standard'):
    true = source_df.label.map(LABEL2ID).to_numpy()
    preds = np.asarray(preds,dtype=int)
    precision,recall,f1,support=precision_recall_fscore_support(
        true,preds,labels=[0,1,2],zero_division=0)
    metrics={
        'model':model_tag,'split':split_name,'prompt_variant':prompt_variant,
        'n':len(true), 'accuracy':float(accuracy_score(true,preds)),
        'macro_f1':float(np.mean(f1)), 'weighted_f1':float(np.average(f1,weights=support)),
        'macro_precision':float(np.mean(precision)), 'macro_recall':float(np.mean(recall)),
        'invalid_outputs':int((preds==-1).sum()),
        'invalid_rate':float((preds==-1).mean())
    }
    for i,label in enumerate(LABELS):
        metrics[f'f1_{label}']=float(f1[i])
        metrics[f'recall_{label}']=float(recall[i])
        metrics[f'precision_{label}']=float(precision[i])
    pred_df=pd.DataFrame({'row_id':source_df.row_id,'truth':source_df.label,
                          'prediction':[LABELS[x] if x>=0 else 'INVALID' for x in preds],
                          'generated_text':raw_outputs})
    fn=f'{split_name}_{model_tag}_{prompt_variant}'
    pred_df.to_csv(OUT/'predictions'/f'{fn}.csv',index=False) # No original financial sentences
    (OUT/'reports'/f'{fn}.json').write_text(json.dumps(metrics,indent=2))
    return metrics,pred_df

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_safetensors=True).to(DEVICE)
baseline_parameters = sum(p.numel() for p in model.parameters())
start = time.perf_counter()
val_preds,val_generated = generate_predictions(model,val_df)
baseline_eval_seconds=time.perf_counter()-start
base_metrics, _=summarise_predictions(val_df,val_preds,val_generated,'baseline','validation')
base_metrics['evaluation_seconds']=baseline_eval_seconds
base_metrics['total_parameters']=baseline_parameters
base_metrics['trainable_parameters']=0
base_metrics['training_seconds']=0
base_metrics['peak_training_gpu_allocated_gib']=0
(OUT/'reports'/'validation_baseline_standard.json').write_text(json.dumps(base_metrics,indent=2))
print(json.dumps(base_metrics,indent=2))
del model;gc.collect();torch.cuda.empty_cache()

def to_dataset(part):
    return Dataset.from_pandas(part[['sentence','label','row_id']],preserve_index=False)
DATA=DatasetDict({'train':to_dataset(train_df),'validation':to_dataset(val_df)})

def encode(batch):
    features=TOKENIZER([prompt(s) for s in batch['sentence']],max_length=CONFIG['input_max_tokens'],
                       truncation=True)
    labels=TOKENIZER(text_target=batch['label'],max_length=CONFIG['target_max_tokens'],truncation=True)
    features['labels']=labels['input_ids']
    return features
TOK_DATA=DATA.map(encode,batched=True,remove_columns=['sentence','label','row_id'])

# Cheap training-path smoke check: labels, tokenizer and real model forward pass.
# It does not update weights or use the locked test set.
_preflight_model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_safetensors=True).to(DEVICE)
_preflight_batch=DataCollatorForSeq2Seq(TOKENIZER,model=_preflight_model,label_pad_token_id=-100)(
    [TOK_DATA['train'][0],TOK_DATA['train'][1]])
_preflight_batch={k:v.to(DEVICE) for k,v in _preflight_batch.items()}
with torch.no_grad():
    _preflight_loss = _preflight_model(**_preflight_batch).loss
assert bool(torch.isfinite(_preflight_loss).item()), 'Non-finite loss in data/model preflight'
print('Pre-training forward-pass check passed; loss =', float(_preflight_loss))
del _preflight_model, _preflight_batch, _preflight_loss
gc.collect(); torch.cuda.empty_cache()

runs=[]
validation_summary=[base_metrics]

def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_variant(tag,kind='full',rank=None,seed=SEED):
    assert kind in ('full','lora')
    if kind=='lora': assert rank and rank>0
    set_seed(seed)
    clean_gpu()
    torch.cuda.reset_peak_memory_stats()
    net=AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_safetensors=True)
    if kind=='lora':
        params=LoraConfig(r=rank,lora_alpha=CONFIG['lora_alpha'],
                          target_modules=['q','v'],lora_dropout=CONFIG['lora_dropout'],
                          bias='none',task_type=TaskType.SEQ_2_SEQ_LM)
        net=get_peft_model(net,params)
    trainable=sum(p.numel() for p in net.parameters() if p.requires_grad)
    total=sum(p.numel() for p in net.parameters())
    run_path=OUT/'models'/tag
    kwargs=dict(
        output_dir=str(run_path/'checkpoints'),
        overwrite_output_dir=True,
        per_device_train_batch_size=CONFIG['train_batch_size'],
        per_device_eval_batch_size=CONFIG['eval_batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_train_epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate_full'] if kind=='full' else CONFIG['learning_rate_lora'],
        weight_decay=CONFIG['weight_decay'], warmup_ratio=CONFIG['warmup_ratio'],
        optim='adamw_torch',
        fp16=False,bf16=BF16,
        eval_strategy='epoch',save_strategy='epoch',
        save_total_limit=1,load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False,
        logging_strategy='steps',logging_steps=20,
        predict_with_generate=False,
        report_to='none',seed=seed,data_seed=SEED,
        push_to_hub=False,dataloader_num_workers=0
    )
    args=Seq2SeqTrainingArguments(**kwargs)
    collator=DataCollatorForSeq2Seq(TOKENIZER,model=net,label_pad_token_id=-100)
    trainer=Seq2SeqTrainer(model=net,args=args,train_dataset=TOK_DATA['train'],
                           eval_dataset=TOK_DATA['validation'],data_collator=collator,
                           processing_class=TOKENIZER,
                           callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    started=time.perf_counter()
    output=trainer.train()
    duration=time.perf_counter()-started
    peak_gib=torch.cuda.max_memory_allocated()/2**30
    model_location=run_path/'best_model'
    trainer.save_model(str(model_location)) # full model for FFT; adapters only for LoRA
    TOKENIZER.save_pretrained(str(model_location))
    if kind=='full':
        assert (model_location/'config.json').is_file(), f'Missing full model configuration: {model_location}'
        assert any(model_location.glob('*.safetensors')), f'Missing full model weights: {model_location}'
    else:
        assert (model_location/'adapter_config.json').is_file(), f'Missing LoRA adapter configuration: {model_location}'
        assert any(model_location.glob('adapter_model*.safetensors')), f'Missing LoRA adapter weights: {model_location}'
    history=pd.DataFrame(trainer.state.log_history)
    history.to_csv(OUT/'logs'/f'{tag}_history.csv',index=False)
    run_meta={
        'tag':tag,'kind':kind,'rank':rank,'seed':seed,
        'trainable_parameters':trainable,'total_parameters':total,
        'trainable_fraction':trainable/total,
        'training_seconds':duration,
        'peak_training_gpu_allocated_gib':float(peak_gib),
        'best_checkpoint':trainer.state.best_model_checkpoint,
        'best_validation_loss':trainer.state.best_metric,
        'best_model_path':str(model_location),
        'train_loss':output.training_loss,
        'global_steps':trainer.state.global_step,
        'config':kwargs,
    }
    (OUT/'reports'/f'{tag}_training.json').write_text(json.dumps(run_meta,indent=2,default=str))
    val_pred,val_txt=generate_predictions(trainer.model,val_df)
    val_metrics,_=summarise_predictions(val_df,val_pred,val_txt,tag,'validation')
    val_metrics.update({k:run_meta[k] for k in ('total_parameters','trainable_parameters',
                                                'training_seconds','peak_training_gpu_allocated_gib',
                                                'best_validation_loss')})
    (OUT/'reports'/f'validation_{tag}_standard.json').write_text(json.dumps(val_metrics,indent=2))
    print(f"{tag}: validation Macro F1={val_metrics['macro_f1']:.4f}, "
          f"training {duration/60:.1f} min, peak allocated VRAM={peak_gib:.2f} GiB")
    del trainer,net,collator,history
    clean_gpu()
    return run_meta,val_metrics

if CONFIG['run_full_finetuning']:
    full_run, full_val = train_variant('full_finetuning',kind='full')
    runs.append(full_run); validation_summary.append(full_val)
else:
    note={'full_finetuning_executed':False,'reason':'MUST SUPPLY MEASURED HARDWARE FEASIBILITY EVIDENCE'}
    (OUT/'reports'/'full_finetuning_exception.json').write_text(json.dumps(note,indent=2))

for rank in CONFIG['lora_ranks']:
    record,measures=train_variant(f'lora_r{rank}',kind='lora',rank=rank)
    runs.append(record);validation_summary.append(measures)

valid_table=pd.DataFrame(validation_summary).sort_values('macro_f1',ascending=False)
valid_table.to_csv(OUT/'reports'/'validation_model_comparison.csv',index=False)
display(valid_table[['model','accuracy','macro_f1','weighted_f1','invalid_rate',
                     'trainable_parameters','training_seconds']])

lora_val=[row for row in validation_summary if row['model'].startswith('lora_r')]
assert len(lora_val)==len(CONFIG['lora_ranks'])
chosen=sorted(lora_val,key=lambda x:(-x['macro_f1'],int(x['model'].split('r')[-1])))[0]
BEST_LORA=chosen['model']
selection={'selection_set':'validation', 'metric':'macro_f1',
           'tie_breaker':'smaller rank','selected_lora':BEST_LORA,
           'ranks_considered':CONFIG['lora_ranks'],
           'validation_macro_f1':{x['model']:x['macro_f1'] for x in lora_val}}
(OUT/'artifacts'/'adapter_selection.json').write_text(json.dumps(selection,indent=2))
print('Selected LoRA via validation:',BEST_LORA)

def load_saved_model(tag):
    if tag=='baseline':
        model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_safetensors=True)
    elif tag=='full_finetuning':
        model=AutoModelForSeq2SeqLM.from_pretrained(OUT/'models'/tag/'best_model')
    elif tag.startswith('lora_'):
        base=AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_safetensors=True)
        model=PeftModel.from_pretrained(base,str(OUT/'models'/tag/'best_model'))
    else:
        raise ValueError(tag)
    model=model.to(DEVICE)
    model.eval()
    return model

FINAL_TAGS=['baseline']+(['full_finetuning'] if CONFIG['run_full_finetuning'] else [])+[BEST_LORA]
final_results=[]
for tag in FINAL_TAGS:
    clean_gpu()
    model=load_saved_model(tag)
    started=time.perf_counter()
    preds,outputs=generate_predictions(model,test_df)
    sec=time.perf_counter()-started
    metrics,_=summarise_predictions(test_df,preds,outputs,tag,'test')
    metrics['test_inference_seconds']=sec
    match=next((x for x in runs if x['tag']==tag),None)
    if match:
        for field in ('total_parameters','trainable_parameters','training_seconds',
                      'peak_training_gpu_allocated_gib','best_validation_loss'):
            metrics[field]=match[field]
    else:
        for field in ('total_parameters','trainable_parameters','training_seconds',
                      'peak_training_gpu_allocated_gib'):
            metrics[field]=base_metrics[field]
    final_results.append(metrics)
    del model;clean_gpu()

final_df=pd.DataFrame(final_results)
final_df.to_csv(OUT/'reports'/'final_test_metrics.csv',index=False)
display(final_df[['model','accuracy','macro_f1','weighted_f1','invalid_rate',
                  'training_seconds','peak_training_gpu_allocated_gib']])

TRUE=test_df.label.map(LABEL2ID).to_numpy()
PRED={}
for tag in FINAL_TAGS:
    csv=pd.read_csv(OUT/'predictions'/f'test_{tag}_standard.csv')
    assert csv.row_id.tolist()==test_df.row_id.tolist()
    PRED[tag]=csv.prediction.map(lambda x:LABEL2ID.get(x,-1)).to_numpy()

rng=np.random.default_rng(SEED+123)
bootstrap_indices=rng.integers(0,len(TRUE),size=(CONFIG['bootstrap_samples'],len(TRUE)))

def bootstrap_f1(pred):
    return np.asarray([f1_score(TRUE[ix],pred[ix],labels=[0,1,2],average='macro',
                                zero_division=0) for ix in bootstrap_indices])

b_samples={tag:bootstrap_f1(PRED[tag]) for tag in FINAL_TAGS}
ci_rows=[]
for tag in FINAL_TAGS:
    ci=np.percentile(b_samples[tag],[2.5,97.5])
    ci_rows.append({'model':tag,'observed_macro_f1':f1_score(TRUE,PRED[tag],labels=[0,1,2],
                                                        average='macro',zero_division=0),
                    'ci95_lower':ci[0],'ci95_upper':ci[1]})
ci_df=pd.DataFrame(ci_rows)
ci_df.to_csv(OUT/'reports'/'macro_f1_bootstrap_ci.csv',index=False)

delta_rows=[]
for i in range(len(FINAL_TAGS)):
    for j in range(i+1,len(FINAL_TAGS)):
        a,b=FINAL_TAGS[i],FINAL_TAGS[j]
        d=b_samples[b]-b_samples[a]
        low,high=np.percentile(d,[2.5,97.5])
        delta_rows.append({'model_a':a,'model_b':b,'delta_b_minus_a':float(np.mean(d)),
                           'ci95_lower':float(low),'ci95_upper':float(high)})
pd.DataFrame(delta_rows).to_csv(OUT/'reports'/'paired_bootstrap_differences.csv',index=False)
display(ci_df)
display(pd.DataFrame(delta_rows))

def save_figure(name):
    plt.tight_layout()
    plt.savefig(OUT/'figures'/f'{name}.png',dpi=250,bbox_inches='tight')
    plt.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight')
    plt.show()

# Main comparison with non-overlapping confidence intervals only as visual uncertainty indicators.
fig,ax=plt.subplots(figsize=(8,4))
x=np.arange(len(ci_df))
point=ci_df.observed_macro_f1.values
lo=point-ci_df.ci95_lower.values;hi=ci_df.ci95_upper.values-point
ax.errorbar(x,point,yerr=np.stack([lo,hi]),fmt='o',capsize=5)
ax.set_xticks(x,ci_df.model.tolist(),rotation=15,ha='right')
ax.set_ylim(0,1)
ax.set_ylabel('Macro F1 on locked test set')
ax.set_title('Held-out macro F1 (95% bootstrap confidence intervals)')
save_figure('02_final_macro_f1_bootstrap')

# Per-class F1 grouped bar plot.
f1cols=[f'f1_{x}' for x in LABELS]
fig,ax=plt.subplots(figsize=(9,4.5))
width=.8/len(FINAL_TAGS)
for i,tag in enumerate(FINAL_TAGS):
    row=final_df.set_index('model').loc[tag]
    ax.bar(np.arange(3)-0.4+(i+.5)*width,[row[c] for c in f1cols],width,label=tag)
ax.set_xticks(np.arange(3),LABELS)
ax.set_ylabel('Class F1');ax.set_ylim(0,1);ax.legend()
ax.set_title('Class-wise F1 by model')
save_figure('03_per_class_f1')

for tag in FINAL_TAGS:
    cm=confusion_matrix(TRUE,PRED[tag],labels=[0,1,2,-1])[:3,:]
    fig,ax=plt.subplots(figsize=(6,4.5))
    ax.imshow(cm,cmap='Blues')
    for rowidx in range(3):
        for colidx in range(4):
            ax.text(colidx,rowidx,str(cm[rowidx,colidx]),ha='center',va='center',
                    color='white' if cm[rowidx,colidx]>cm.max()/2 else 'black')
    ax.set_xticks(range(4),LABELS+['INVALID'],rotation=25)
    ax.set_yticks(range(3),LABELS)
    ax.set_xlabel('Generated prediction');ax.set_ylabel('Reference')
    ax.set_title(f'Confusion matrix: {tag}')
    save_figure('04_confusion_'+tag)

# Train curves from actual trainer logs.
fig,ax=plt.subplots(figsize=(8,4))
for rec in runs:
    h=pd.read_csv(OUT/'logs'/f"{rec['tag']}_history.csv")
    train=h[['step','loss']].dropna()
    if len(train):ax.plot(train.step,train.loss,label=rec['tag'])
ax.set_xlabel('Optimiser steps');ax.set_ylabel('Training loss')
ax.set_title('Training loss by adaptation method');ax.legend()
save_figure('05_training_loss')

# Compute efficiency: no inference-only baseline on training-time graph.
train_compare=final_df[final_df.model!='baseline']
fig,ax=plt.subplots(figsize=(7,4.5))
for _,row in train_compare.iterrows():
    ax.scatter(row.training_seconds/60,row.macro_f1,s=90)
    ax.annotate(row.model,(row.training_seconds/60,row.macro_f1),xytext=(5,5),textcoords='offset points')
ax.set_xlabel('Fine-tuning elapsed time (minutes)');ax.set_ylabel('Test macro F1')
ax.set_title('Performance versus training time')
save_figure('06_efficiency')

# Error taxonomy is model-independent and reproducible.
error_rows=[]
for tag in FINAL_TAGS:
    frame=pd.read_csv(OUT/'predictions'/f'test_{tag}_standard.csv')
    wrong=frame[frame.truth!=frame.prediction].copy()
    wrong['error_type']=np.where(wrong.prediction=='INVALID','invalid_generation',
                                 wrong.truth+'__to__'+wrong.prediction)
    wrong['model']=tag
    error_rows.append(wrong)
errors=pd.concat(error_rows,ignore_index=True)
errors.to_csv(OUT/'reports'/'misclassifications_by_hash.csv',index=False)
errors.groupby(['model','error_type']).size().rename('count').reset_index().to_csv(
    OUT/'reports'/'error_type_counts.csv',index=False)
display(errors.groupby(['model','error_type']).size().rename('count').reset_index())

robust_rows=[]
for tag in FINAL_TAGS:
    model=load_saved_model(tag)
    p,texts=generate_predictions(model,test_df,prompt_variant='alternative')
    m,_=summarise_predictions(test_df,p,texts,tag,'test','alternative')
    primary=next(x for x in final_results if x['model']==tag)
    robust_rows.append({'model':tag,'primary_macro_f1':primary['macro_f1'],
                        'alternative_macro_f1':m['macro_f1'],
                        'delta_alternative_minus_primary':m['macro_f1']-primary['macro_f1'],
                        'primary_invalid_rate':primary['invalid_rate'],
                        'alternative_invalid_rate':m['invalid_rate']})
    del model;clean_gpu()
robust_df=pd.DataFrame(robust_rows)
robust_df.to_csv(OUT/'reports'/'prompt_sensitivity.csv',index=False)
display(robust_df)

assert set(final_df.model)==set(FINAL_TAGS)
assert all((final_df.accuracy>=0)&(final_df.accuracy<=1))
assert all((final_df.macro_f1>=0)&(final_df.macro_f1<=1))
assert not any(pd.isna(final_df.macro_f1))
assert (OUT/'artifacts'/'adapter_selection.json').is_file()

CARD=f"""# Model card — 7043SCN Task 1

**Student:** {STUDENT['name']} ({STUDENT['student_id']})  
**Application:** English financial news sentiment classification  
**Base model:** [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) — encoder–decoder text-to-text transformer  
**Data source:** Financial PhraseBank (Malo et al., 2014), Sentences_{CONFIG['agreement']}Agree, downloaded from [{SOURCE_URL}]({SOURCE_URL})  
**Dataset licence:** CC BY-NC-SA 3.0; maintain attribution and non-commercial licence conditions.  
**Classes:** negative / neutral / positive  
**Data partitions:** train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}.  
**Selection:** LoRA rank selected using validation macro-F1; final test was reserved for one post-selection evaluation.  
**LoRA rank selected:** {BEST_LORA}.  
**Hardware:** {GPU_NAME} ({GPU_GB:.2f} GiB).  
**Seed:** {SEED}.  

## Methods
Pretrained zero-update baseline; all-parameter supervised fine-tuning; LoRA with q/v target modules at ranks {CONFIG['lora_ranks']}. Training checkpoint selection used validation loss. See runtime_config.json, training reports and learning curves.

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
"""
(OUT/'artifacts'/'MODEL_CARD.md').write_text(CARD)
summary={'experiment_status':'complete_three_way' if CONFIG['run_full_finetuning'] else 'partial_hardware_exception', 'student':STUDENT,
         'source_sha256':source_sha256,'run_count':len(runs),
         'selected_lora':BEST_LORA,'final_test_models':FINAL_TAGS,
         'result_paths':['reports/final_test_metrics.csv','reports/prompt_sensitivity.csv',
                         'reports/macro_f1_bootstrap_ci.csv']}
(OUT/'artifacts'/'experiment_summary.json').write_text(json.dumps(summary,indent=2))

# Exact output audit before creating a shareable results archive.
EXPECTED_RESULTS = [
    'artifacts/runtime_config.json','artifacts/source_manifest.json',
    'artifacts/split_manifest.csv','artifacts/split_class_counts.csv',
    'artifacts/adapter_selection.json','artifacts/MODEL_CARD.md',
    'artifacts/experiment_summary.json','reports/validation_model_comparison.csv',
    'reports/final_test_metrics.csv','reports/macro_f1_bootstrap_ci.csv',
    'reports/paired_bootstrap_differences.csv','reports/prompt_sensitivity.csv',
    'reports/error_type_counts.csv','figures/02_final_macro_f1_bootstrap.png',
    'figures/03_per_class_f1.png','figures/05_training_loss.png',
    'figures/06_efficiency.png',
]
for _tag in FINAL_TAGS:
    EXPECTED_RESULTS += [f'predictions/test_{_tag}_standard.csv',
                         f'figures/04_confusion_{_tag}.png']
for _rec in runs:
    EXPECTED_RESULTS += [f'reports/{_rec["tag"]}_training.json',
                         f'logs/{_rec["tag"]}_history.csv']
missing_results = [x for x in EXPECTED_RESULTS if not (OUT/x).is_file()]
assert not missing_results, f'Missing outputs before export: {missing_results}'
assert len(final_df) == len(FINAL_TAGS), 'Not all final models evaluated'
assert (final_df['n'] == len(test_df)).all(), 'Some test rows are missing'
print('FINAL AUDIT PASSED:', len(EXPECTED_RESULTS), 'required files present;', len(final_df), 'models evaluated')

# Portable archive: important plots, metrics and manifests, excluding copyrighted raw texts,
# source ZIP, model weights and training checkpoints.
ZIP_RESULTS=OUT/'Sampath_17522555_Task1_GitHubSafe_Results.zip'
with zipfile.ZipFile(ZIP_RESULTS,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for folder in ('artifacts','reports','figures','logs','predictions'):
        for file in (OUT/folder).rglob('*'):
            if file.is_file(): z.write(file,file.relative_to(OUT))
print('Final models:',FINAL_TAGS)
print('Completed reports:',len(list((OUT/'reports').glob('*'))))
print('Saved:',ZIP_RESULTS)
print('Result ZIP size (MiB):',round(ZIP_RESULTS.stat().st_size/2**20,2))
print('Please download the results ZIP and keep a separate, private backup of model checkpoints.')

try:
    from google.colab import files
    files.download(str(ZIP_RESULTS))
except ImportError:
    print('Not in Colab: retrieve the ZIP from', ZIP_RESULTS)

# Optional private Google Drive backup: uncomment and run in Colab.
# from google.colab import drive
# drive.mount('/content/drive')
# import shutil
# shutil.copytree(OUT, '/content/drive/MyDrive/Sampath_17522555_Task1', dirs_exist_ok=True)
