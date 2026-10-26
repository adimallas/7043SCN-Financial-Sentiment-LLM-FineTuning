"""Check real Task 1 exported measurements without re-running GPU training."""
from pathlib import Path
import json
import math
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'


def check():
    metrics=pd.read_csv(R/'reports'/'final_test_metrics.csv')
    splits=pd.read_csv(R/'artifacts'/'split_manifest.csv')
    assert len(splits)==3448, f'Unexpected split count {len(splits)}'
    assert not splits.row_id.duplicated().any(), 'Split IDs are not unique'
    sizes=splits['split'].value_counts().to_dict()
    assert sizes == {'train':2412,'validation':518,'test':518}, sizes
    expected={'baseline','full_finetuning','lora_r16'}
    assert set(metrics.model)==expected, set(metrics.model)
    test_ids=set(splits.loc[splits.split=='test','row_id'])
    for tag in sorted(expected):
        pred=pd.read_csv(R/'predictions'/f'test_{tag}_standard.csv')
        assert len(pred)==518 and set(pred.row_id)==test_ids, f'Missing or changed predictions: {tag}'
        assert not pred.row_id.duplicated().any(), f'Duplicate predictions: {tag}'
        row=metrics.loc[metrics.model==tag].iloc[0]
        acc=accuracy_score(pred.truth,pred.prediction)
        f1=f1_score(pred.truth,pred.prediction,average='macro',labels=['negative','neutral','positive'],zero_division=0)
        assert math.isclose(acc,float(row.accuracy),abs_tol=1e-12), tag
        assert math.isclose(f1,float(row.macro_f1),abs_tol=1e-12), tag
        print(f'{tag:17s} n={len(pred):3d}  accuracy={acc:.6f}  macro_F1={f1:.6f}')
    source=json.loads((R/'artifacts'/'source_manifest.json').read_text())
    assert len(source['archive_sha256'])==64
    selection=json.loads((R/'artifacts'/'adapter_selection.json').read_text())
    assert selection['selected_lora']=='lora_r16'
    print('PASS: split disjointness, saved predictions, recorded metrics, provenance and adapter selection.')

if __name__=='__main__':check()
