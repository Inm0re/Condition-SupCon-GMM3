"""Portable fitting, calibration and scoring for the public primary cohorts."""
from pathlib import Path
import csv,json
import joblib
import numpy as np
import torch
from . import core,legacy_training
from .encoder import PatchEncoder

VARIANTS=json.loads((Path(__file__).parents[1]/'configs/variants.json').read_text())
PUBLIC_DATASETS=('hust','paderborn_real','seu_dds')

def configure(config,device=None):
    for mod in (core,legacy_training):
        mod.ITERS=int(config['steps']);mod.TEMPERATURE=float(config['temperature'])
        mod.GMM_COMPONENTS=int(config['components']);mod.Q_THRESHOLD=float(config['normal_quantile'])
        mod.PER_CONDITION=int(config.get('anchors_per_condition',4))
        mod.DEVICE=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    # These functions were copied without changing their mathematics. The original
    # contrastive default argument needs explicit rebinding for a requested sweep.
    original=core.supcon_loss
    if hasattr(original,'_unbound'):original=original._unbound
    def bound(features,labels):return original(features,labels,temperature=core.TEMPERATURE)
    bound._unbound=original;core.supcon_loss=bound

def npy(folder,name):return np.load(Path(folder)/name,mmap_mode='r',allow_pickle=False)

def groups(folder,split,n):
    for suffix in ('file_id','bearing','run_id'):
        p=Path(folder)/f'{split}_{suffix}.npy'
        if p.exists():return np.asarray(np.load(p,allow_pickle=False),np.int64)
    return np.zeros(n,np.int64)

def load_normal(folder,dataset):
    folder=Path(folder)
    if dataset=='hust':
        x=npy(folder,'train.npy');c=np.asarray(npy(folder,'train_condition.npy'),np.int64)
        fit=[];cal=[]
        for condition in sorted(np.unique(c)):
            ids=np.flatnonzero(c==condition);start=len(ids)-int(round(.2*len(ids)))
            if start<=7:raise ValueError('Insufficient HUST fitting support for seven guard starts')
            fit.extend(ids[:start-7]);cal.extend(ids[start:])
        return dict(fit_x=x,fit_cond=c,fit_group=groups(folder,'train',len(x)),fit_ids=np.asarray(fit),val_x=x,val_cond=c,val_ids=np.asarray(cal))
    x=npy(folder,'train_x.npy');v=npy(folder,'val_x.npy')
    return dict(fit_x=x,fit_cond=np.asarray(npy(folder,'train_condition.npy'),np.int64),fit_group=groups(folder,'train',len(x)),fit_ids=np.arange(len(x)),val_x=v,val_cond=np.asarray(npy(folder,'val_condition.npy'),np.int64),val_ids=np.arange(len(v)))

def ensure_supported(conditions,supported):
    absent=set(map(int,np.unique(conditions)))-set(map(int,supported))
    if absent:raise ValueError(f'Unsupported operating-condition identifiers: {sorted(absent)}')

def raw_score(z,conditions,banks,family,routing):
    conditions=np.asarray(conditions,np.int64);result=np.empty(len(z),np.float32)
    ensure_supported(conditions,banks.keys() if routing=='condition' else banks['global']['conditions'])
    for key,bank in banks.items():
        rows=np.ones(len(z),bool) if routing=='global' else conditions==int(key)
        if not rows.any():continue
        if bank.get('score_policy')=='lifecycle_slogdet':
            from .lifecycle import lifecycle_component_score
            result[rows]=lifecycle_component_score(z[rows],bank)
        else:result[rows]=core.gmm_nll(z[rows],bank['models'],bank['priors']) if family=='gmm' else core.knn_scores(z[rows],bank['neighbors'])
    return result

def fit_banks(z,c,seed,family,routing):
    banks={};keys=sorted(np.unique(c)) if routing=='condition' else ['global']
    budget=core.total_bank_budget(len(z),len(np.unique(c)))
    allocations=core.allocate_condition_budgets(c,budget) if routing=='condition' else {'global':budget}
    for key in keys:
        local=z[c==key] if routing=='condition' else z
        local_seed=seed+100*int(key) if routing=='condition' else seed
        if family=='gmm':
            models,priors,counts=core.fit_gmm(local,local_seed,core.GMM_COMPONENTS)
            bank=dict(models=models,priors=priors,counts=counts)
        else:bank=dict(neighbors=core.make_coreset(local,local_seed,allocations[key]))
        bank['conditions']=list(map(int,np.unique(c)));banks[str(key)]=bank
    return banks

def align(raw,conditions,scales,routing):
    result=np.empty(len(raw),np.float32)
    ensure_supported(conditions,scales.keys() if routing=='condition' else scales['global']['conditions'])
    for key,scale in scales.items():
        rows=np.ones(len(raw),bool) if routing=='global' else conditions==int(key)
        result[rows]=(np.asarray(raw[rows],np.float64)-scale['median'])/scale['iqr']
    return result

def fit(folder,output,config,seed,variant='H',device=None):
    configure(config,device);dataset=config['dataset']
    if dataset not in PUBLIC_DATASETS:raise ValueError('Use the lifecycle entry point for XJTU-SY')
    spec=next(v for v in VARIANTS if v['variant'].split('_')[0]==variant)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    normal=load_normal(folder,dataset);c=normal['fit_cond'][normal['fit_ids']]
    ensure_supported(normal['val_cond'][normal['val_ids']],np.unique(c))
    if normal['fit_x'].shape[-1]!=config['channels']:raise ValueError('Unexpected input channels')
    mean,std=core.normalization(normal['fit_x'],normal['fit_cond'],normal['fit_ids'])
    if config['training_backend']=='legacy':
        model=legacy_training.fit_encoder(spec['encoder'],normal,mean,std,seed,output)
    else:model=core.fit_encoder(spec['encoder'],normal,mean,std,seed,output)
    z=core.embed(model,normal['fit_x'],normal['fit_cond'],normal['fit_ids'],mean,std)
    banks=fit_banks(z,c,seed,spec['family'],spec['routing'])
    zv=core.embed(model,normal['val_x'],normal['val_cond'],normal['val_ids'],mean,std)
    cv=normal['val_cond'][normal['val_ids']];rv=raw_score(zv,cv,banks,spec['family'],spec['routing'])
    scales={}
    for key in sorted(np.unique(cv)) if spec['routing']=='condition' else ['global']:
        local=rv[cv==key] if spec['routing']=='condition' else rv
        med,iqr=core.robust_scale(np.asarray(local,np.float64))
        scales[str(key)]=dict(median=med,iqr=iqr,conditions=list(map(int,np.unique(c))))
    threshold=float(np.quantile(align(rv,cv,scales,spec['routing']),config['normal_quantile']))
    bundle=dict(mean=mean,std=std,banks=banks,scales=scales,threshold=threshold,spec=spec,config=config,seed=seed)
    joblib.dump(bundle,output/'normal_reference.joblib')
    core.dump(dict(dataset=dataset,variant=variant,seed=seed,fit_windows=len(c),calibration_windows=len(cv),threshold=threshold,test_data_opened=False),output/'fit_protocol.json')
    return output

def predict(x,conditions,artifact,device=None):
    artifact=Path(artifact);bundle=joblib.load(artifact/'normal_reference.joblib')
    configure(bundle['config'],device);c=np.asarray(conditions,np.int64)
    ensure_supported(c,[int(k) for k in bundle['scales']] if bundle['spec']['routing']=='condition' else bundle['scales']['global']['conditions'])
    model=PatchEncoder(in_channels=x.shape[-1],projection_dim=256,use_revin=False).to(core.DEVICE)
    model.load_state_dict(torch.load(artifact/'model.pt',map_location=core.DEVICE,weights_only=True))
    z=core.embed(model,x,c,np.arange(len(x)),bundle['mean'],bundle['std'])
    raw=raw_score(z,c,bundle['banks'],bundle['spec']['family'],bundle['spec']['routing'])
    score=align(raw,c,bundle['scales'],bundle['spec']['routing'])
    threshold=np.asarray([bundle['threshold'][str(int(k))] for k in c]) if isinstance(bundle['threshold'],dict) else bundle['threshold']
    return score,(score>=threshold).astype(np.int8),threshold

def metrics(y,score,prediction):
    y=np.asarray(y,np.int8);pred=np.asarray(prediction,np.int8)
    tn,fp,fn,tp=core.confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(auroc=float(core.roc_auc_score(y,score)),auprc=float(core.average_precision_score(y,score)),f1=float(core.f1_score(y,pred,zero_division=0)),mcc=float(core.matthews_corrcoef(y,pred)),normal_fpr=float(fp/(tn+fp)),tpr=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),balanced_accuracy=float((tn/(tn+fp)+tp/(tp+fn))/2),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def evaluate(folder,artifact,dataset,device=None):
    x=npy(folder,'test.npy' if dataset=='hust' else 'test_x.npy');c=npy(folder,'test_condition.npy')
    scores,pred,threshold=predict(x,c,artifact,device)
    # Labels are opened only after frozen scoring.
    y=npy(folder,'test_label.npy')
    selected=np.arange(len(y))
    if dataset=='paderborn_real':
        fault=npy(folder,'test_fault_type.npy')
        selected=np.flatnonzero((y==0)|np.isin(fault,[3,4]))
    result=metrics(y[selected],scores[selected],pred[selected]);result['threshold']=threshold
    np.save(Path(artifact)/'test_score.npy',scores[selected])
    np.save(Path(artifact)/'test_source_indices.npy',selected)
    core.dump(result,Path(artifact)/'metrics.json')
    return result
