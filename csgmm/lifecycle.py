"""The two-bearing chronological XJTU-SY protocol; no pretrained artifacts shipped."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from . import core,lifecycle_training
from .pipeline import predict_raw,metrics,ensure_supported
from .lifecycle_scoring import aggregate_records,fit_record_calibration,apply_record_calibration

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,nargs='+',default=list(range(35001,35006)))
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--device',choices=['cpu','cuda'])
    a=p.parse_args()
    normal={s:(np.load(a.data/f'{s}_x.npy',mmap_mode='r'),np.load(a.data/f'{s}_condition.npy')) for s in ['train','val']}
    x,c=normal['train'];v,cv=normal['val'];ensure_supported(cv,np.unique(c))
    cfg=json.loads((Path(__file__).parents[1]/'configs/xjtu_sy.json').read_text());cfg['steps']=a.steps
    from .pipeline import configure
    configure(cfg,a.device)
    for seed in a.seed:
        out=a.output/f'seed_{seed}'/'H';out.mkdir(parents=True,exist_ok=False)
        mean=np.stack([np.asarray(x[c==k],np.float32).mean((0,1)) for k in range(2)])
        std=np.stack([np.maximum(np.asarray(x[c==k],np.float32).std((0,1)),1e-5) for k in range(2)])
        lifecycle_training.SEED=seed;lifecycle_training.ITERS=a.steps;lifecycle_training.DEVICE=core.DEVICE
        model,_=lifecycle_training.fit_variant('supcon_condition',x,c,mean,std,out)
        z=core.embed(model,x,c,np.arange(len(x)),mean,std);zv=core.embed(model,v,cv,np.arange(len(v)),mean,std)
        banks=fit_lifecycle_banks(z,c);calibration_raw=np.empty(len(v),np.float32)
        for condition in range(2):
            calibration_raw[cv==condition]=lifecycle_component_score(zv[cv==condition],banks[str(condition)])
        # V48 calls V31: aggregate NORMAL windows before estimating calibration.
        calibration_records=aggregate_records(calibration_raw,cv,np.load(a.data/'val_record.npy'))
        scales,thresholds=fit_record_calibration(calibration_records)
        bundle=dict(mean=mean,std=std,banks=banks,scales=scales,threshold=thresholds,spec=dict(family='gmm',routing='condition'),config=cfg,seed=seed,score_unit='record',record_aggregation=.90,calibration_protocol='V31_V48_record_q90_median_iqr_q95')
        joblib.dump(bundle,out/'normal_reference.joblib')
        core.dump(dict(dataset='xjtu_sy',seed=seed,fit_windows=len(x),calibration_windows=len(v),calibration_records=len(calibration_records['raw']),record_aggregation=.9,threshold_scale='median-IQR calibrated normal record q90',calibration=scales,thresholds=thresholds,test_data_opened=False),out/'fit_protocol.json')
        test=np.load(a.data/'test_x.npy',mmap_mode='r');ct=np.load(a.data/'test_condition.npy')
        raw,_=predict_raw(test,ct,out,a.device)
        record=np.load(a.data/'test_record.npy');y=np.load(a.data/'test_label.npy')
        test_records=aggregate_records(raw,ct,record)
        scores,predictions,_=apply_record_calibration(test_records,scales,thresholds)
        rows=[]
        for condition in range(2):
            group=np.flatnonzero(test_records['condition']==condition);ys=[]
            ids=test_records['record'][group];rs=scores[group]
            for rid in ids:
                selected=np.flatnonzero((ct==condition)&(record==rid))
                if not np.all(y[selected]==y[selected[0]]):raise ValueError('A record contains inconsistent labels')
                ys.append(int(y[selected[0]]))
            result=metrics(ys,rs,predictions[group])
            result.update(dataset='xjtu_sy',bearing='Bearing1-1' if condition==0 else 'Bearing2-2',seed=seed,threshold=thresholds[str(condition)])
            rows.append(result)
            np.savez(out/f'bearing_{condition}_record_scores.npz',record=ids,raw_score=test_records['raw'][group],score=rs,label=ys,threshold=thresholds[str(condition)])
        core.dump(rows,out/'record_metrics.json');print(json.dumps(rows,indent=2))

def fit_lifecycle_banks(z,conditions):
    banks={}
    for condition in sorted(np.unique(conditions)):
        local=z[conditions==condition]
        km=MiniBatchKMeans(n_clusters=3,random_state=100+int(condition),batch_size=512,max_iter=100,n_init=5).fit(local)
        models=[];priors=[]
        for cluster in range(3):
            member=local[km.labels_==cluster]
            if len(member)<2:raise ValueError('Insufficient component support in lifecycle fitting')
            models.append(LedoitWolf().fit(member));priors.append(max(len(member)/len(local),1e-8))
        banks[str(int(condition))]=dict(models=models,priors=priors,conditions=list(map(int,np.unique(conditions))),score_policy='lifecycle_slogdet')
    return banks

def lifecycle_component_score(z,bank):
    values=[]
    for model,prior in zip(bank['models'],bank['priors']):
        sign,det=np.linalg.slogdet(model.covariance_)
        if sign<=0:raise ValueError('Non-positive covariance determinant')
        values.append(.5*(core.quadratic(z,model.location_,model.precision_)+float(det))-np.log(prior))
    return np.min(np.stack(values,1),1).astype(np.float32)

if __name__=='__main__':main()
