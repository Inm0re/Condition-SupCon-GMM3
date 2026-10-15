"""Lifecycle inference on complete records using normal-record calibration."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from .pipeline import predict_raw
from .lifecycle_scoring import aggregate_records,apply_record_calibration


def predict_records(windows,conditions,record_ids,artifact,device=None):
    bundle=joblib.load(Path(artifact)/'normal_reference.joblib')
    if bundle.get('calibration_protocol')!='V31_V48_record_q90_median_iqr_q95':
        raise ValueError('Expected corrected record-calibrated lifecycle artifacts (v1.0.1+); v1.0.0 artifacts are incompatible')
    raw,bundle=predict_raw(windows,conditions,artifact,device)
    records=aggregate_records(raw,conditions,record_ids)
    score,prediction,threshold=apply_record_calibration(records,bundle['scales'],bundle['threshold'])
    return dict(condition=records['condition'],record=records['record'],raw_score=records['raw'],score=score,prediction=prediction,threshold=threshold)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--windows',type=Path,required=True)
    p.add_argument('--conditions',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'])
    a=p.parse_args()
    result=predict_records(np.load(a.windows,mmap_mode='r'),np.load(a.conditions),np.load(a.records),a.artifact,a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.output,**result)


if __name__=='__main__':main()
