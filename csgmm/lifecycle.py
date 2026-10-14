"""The two-bearing chronological XJTU-SY protocol; no pretrained artifacts shipped."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from . import core,lifecycle_training
from .pipeline import predict,metrics,ensure_supported

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
        banks={};thresholds={}
        for condition in range(2):
            local=z[c==condition]
            km=MiniBatchKMeans(n_clusters=3,random_state=100+condition,batch_size=512,max_iter=100,n_init=5).fit(local)
            models=[];priors=[]
            for cluster in range(3):
                member=local[km.labels_==cluster]
                if len(member)<2:raise ValueError('Insufficient component support in lifecycle fitting')
                models.append(LedoitWolf().fit(member));priors.append(max(len(member)/len(local),1e-8))
            banks[str(condition)]=dict(models=models,priors=priors,conditions=[0,1],score_policy='lifecycle_slogdet')
            scores=lifecycle_component_score(zv[cv==condition],banks[str(condition)])
            # Archived policy: threshold on normal calibration WINDOW scores,
            # decisions on q90 record scores. This intentionally remains explicit.
            thresholds[str(condition)]=float(np.quantile(scores,.95))
        bundle=dict(mean=mean,std=std,banks=banks,scales={str(c):dict(median=0.,iqr=1.,conditions=[0,1]) for c in range(2)},threshold=thresholds,spec=dict(family='gmm',routing='condition'),config=cfg,seed=seed)
        joblib.dump(bundle,out/'normal_reference.joblib')
        core.dump(dict(dataset='xjtu_sy',seed=seed,fit_windows=len(x),calibration_windows=len(v),record_aggregation=.9,threshold_scale='normal calibration windows',test_data_opened=False),out/'fit_protocol.json')
        test=np.load(a.data/'test_x.npy',mmap_mode='r');ct=np.load(a.data/'test_condition.npy')
        scores,_,_=predict(test,ct,out,a.device)
        record=np.load(a.data/'test_record.npy');y=np.load(a.data/'test_label.npy')
        rows=[]
        for condition in range(2):
            group=np.flatnonzero(ct==condition);rs=[];ys=[];ids=[]
            for rid in np.unique(record[group]):
                selected=group[record[group]==rid];assert len(selected)==16
                rs.append(float(np.quantile(scores[selected],.9)));ys.append(int(y[selected[0]]));ids.append(int(rid))
            result=metrics(ys,rs,np.asarray(rs)>=thresholds[str(condition)])
            result.update(dataset='xjtu_sy',bearing='Bearing1-1' if condition==0 else 'Bearing2-2',seed=seed,threshold=thresholds[str(condition)])
            rows.append(result)
            np.savez(out/f'bearing_{condition}_record_scores.npz',record=ids,score=rs,label=ys)
        core.dump(rows,out/'record_metrics.json');print(json.dumps(rows,indent=2))

def lifecycle_component_score(z,bank):
    values=[]
    for model,prior in zip(bank['models'],bank['priors']):
        sign,det=np.linalg.slogdet(model.covariance_)
        if sign<=0:raise ValueError('Non-positive covariance determinant')
        values.append(.5*(core.quadratic(z,model.location_,model.precision_)+float(det))-np.log(prior))
    return np.min(np.stack(values,1),1).astype(np.float32)

if __name__=='__main__':main()
