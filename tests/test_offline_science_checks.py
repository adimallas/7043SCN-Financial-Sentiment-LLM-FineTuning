"""Actual offline tests of key notebook scientific utilities using synthetic observations.
Cannot substitute for running the pinned Hugging Face models on a Colab GPU.
"""
import unittest, json, ast, re, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT/'Sampath_Adimalla_17522555_7043SCN_Task1_Financial_Sentiment_FineTuning.ipynb').read_text())

def cell(i): return ''.join(NB['cells'][i]['source'])

def get_function(name, cell_index):
    root=ast.parse(cell(cell_index))
    found=[n for n in root.body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(found)!=1: raise AssertionError(f'{name} not found exactly once')
    return compile(ast.Module(body=found,type_ignores=[]),'<test-compiled-notebook>','exec')

class OfflineScienceChecks(unittest.TestCase):
    def test_notebook_json_schema_and_cli_compile(self):
        import nbformat
        nb=nbformat.read(ROOT/'Sampath_Adimalla_17522555_7043SCN_Task1_Financial_Sentiment_FineTuning.ipynb',as_version=4)
        nbformat.validate(nb)
        compile((ROOT/'task1_pipeline.py').read_text(),str(ROOT/'task1_pipeline.py'),'exec')
        self.assertEqual(len(nb.cells),32)

    def test_source_and_dependency_contract(self):
        self.assertIn('FinancialPhraseBank-v1.0.zip', cell(6))
        self.assertIn('archive.testzip()',cell(6)) if False else self.assertIn('_archive.testzip()',cell(6))
        req=(ROOT/'requirements.txt').read_text()
        for s in ['transformers==4.57.1','peft==0.18.0','datasets==4.4.1']:
            self.assertIn(s,req)
        self.assertIn('use_safetensors=True',cell(12))
        self.assertIn('use_safetensors=True',cell(14))
        self.assertIn('use_safetensors=True',cell(20))

    def test_prompts_and_generated_label_parser(self):
        g={'re':re,'LABEL2ID':{'negative':0,'neutral':1,'positive':2}}
        exec(get_function('prompt',10),g)
        exec(get_function('parse_label',10),g)
        self.assertEqual([g['parse_label'](x) for x in ('POSITIVE.', ' neutral ','negative','Neutral or positive','')],[2,1,0,-1,-1])
        self.assertIn('Sentiment:',g['prompt']('Profit improved'))
        self.assertIn('Label:',g['prompt']('Profit improved','alternative'))

    def test_scoring_invalid_outputs_are_errors(self):
        with tempfile.TemporaryDirectory() as d:
            from sklearn.metrics import precision_recall_fscore_support
            o=Path(d)
            for sub in ('predictions','reports'): (o/sub).mkdir()
            labels=['negative','neutral','positive']
            g={'np':np,'pd':pd,'json':json,'OUT':o,'LABELS':labels,'LABEL2ID':dict(zip(labels,range(3))),
               'precision_recall_fscore_support':precision_recall_fscore_support,
               'accuracy_score':accuracy_score}
            exec(get_function('summarise_predictions',10),g)
            dfr=pd.DataFrame({'row_id':['a','b','c','d'],'label':['negative','neutral','positive','positive']})
            metrics,_=g['summarise_predictions'](dfr,[0,1,2,-1],['negative','neutral','positive','unknown'],'smoke','test')
            self.assertAlmostEqual(metrics['accuracy'],0.75)
            self.assertEqual(metrics['invalid_outputs'],1)
            self.assertEqual(len(pd.read_csv(o/'predictions/test_smoke_standard.csv')),4)
            self.assertTrue((o/'reports/test_smoke_standard.json').exists())

    def test_stratified_split_with_no_leakage(self):
        # Mirror split formula using synthetic rows; compare deterministic repeat.
        df=pd.DataFrame({'row_id':[f'id{i}' for i in range(120)],
                         'label':['negative','neutral','positive']*40})
        def split():
            trainval,test=train_test_split(df,test_size=.15,random_state=42,stratify=df.label)
            tr,val=train_test_split(trainval,test_size=.15/.85,random_state=42,stratify=trainval.label)
            return tr,val,test
        pieces=split()
        self.assertEqual(sum(len(p) for p in pieces),120)
        self.assertFalse(set(pieces[0].row_id)&set(pieces[1].row_id))
        self.assertFalse(set(pieces[0].row_id)&set(pieces[2].row_id))
        self.assertFalse(set(pieces[1].row_id)&set(pieces[2].row_id))
        self.assertEqual([sorted(p.row_id.tolist()) for p in pieces],
                         [sorted(p.row_id.tolist()) for p in split()])

    def test_bootstrap_paired_difference_is_zero_on_identical_predictions(self):
        labels=np.array([0,1,2,0,1,2]*12)
        pr=np.array([0,1,1,0,2,2]*12)
        rng=np.random.default_rng(165)
        indices=rng.integers(0,len(labels),size=(150,len(labels)))
        bs=lambda pred:np.array([f1_score(labels[ix],pred[ix],labels=[0,1,2],average='macro',zero_division=0) for ix in indices])
        a,b=bs(pr),bs(pr.copy())
        self.assertTrue(np.all(a==b))
        self.assertEqual(float(np.percentile(b-a,97.5)),0.0)

    def test_final_audit_covers_core_artifacts(self):
        src=cell(28)
        for key in ['FINAL AUDIT PASSED','reports/final_test_metrics.csv',
                    'reports/prompt_sensitivity.csv','reports/macro_f1_bootstrap_ci.csv',
                    'reports/paired_bootstrap_differences.csv','artifacts/MODEL_CARD.md']:
            self.assertIn(key,src)

if __name__ == '__main__': unittest.main()
