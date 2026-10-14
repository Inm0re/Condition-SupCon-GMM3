"""Six prespecified normal-bearing holdouts; fold-only preprocessing; no test tuning."""
from pathlib import Path
import os,json,tempfile,subprocess,hashlib,time,shutil
import numpy as np
import pandas as pd
import torch,joblib
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix

OUT=Path(os.environ['CSGMM_OUTPUT']); OUT.mkdir(parents=True,exist_ok=False)
RAW=Path(os.environ['CSGMM_RAW'])
DATA=Path(os.environ['CSGMM_DATA'])
from . import core as v
V45=Path(v.__file__)
v.EMBED_CHUNK=256; v.ITERS=200; SEED=35001
manifest=pd.read_csv(DATA/'file_manifest.csv'); normal=manifest[manifest.label==0].sort_values('file_id'); faulty=manifest[manifest.fault_type.isin([3,4])].sort_values('file_id')
healthy=[f'K00{i}' for i in range(1,7)]; faultcodes=sorted(faulty.bearing_code.unique()); assert len(faultcodes)==14
def metadata(m):
    n=[]
    for r in m.itertuples():
        for window in range(int(r.window_count)):
            n.append({'bearing':r.bearing_code,'condition':int(r.condition_id),'file_id':int(r.file_id),'source_mat':r.source_mat,'window_in_file':window,'sample_start':2048*window})
    return pd.DataFrame(n)
nm=metadata(normal); fm=metadata(faulty); nm.to_csv(OUT/'normal_window_manifest.csv',index=False); fm.to_csv(OUT/'fault_window_manifest.csv',index=False)
normal.to_csv(OUT/'normal_source_manifest.csv',index=False); faulty.to_csv(OUT/'fault_source_manifest.csv',index=False)
folds=[{'fold':i+1,'test_normal':b,'cal_normal':healthy[(i+1)%6],'fit_normal':[z for z in healthy if z not in [b,healthy[(i+1)%6]]]} for i,b in enumerate(healthy)]
(OUT/'PROTOCOL.json').write_text(json.dumps({'seed':SEED,'folds':folds,'hyperparameters':{'steps':200,'temperature':.1,'K':3,'threshold_q':.95,'per_condition_anchors':4,'learning_rate':1e-4,'weight_decay':1e-4},'preprocessing':'raw unnormalised MAT channels; each fold computes per-condition channel mean/std from its fitting bearings only','source_channels':['phase_current_1','phase_current_2','vibration_1'],'source_support':'first 122 complete 2048-sample nonoverlapping windows, uniform N15_M01_F10 repetition2 exclusion','training':'normal only; minimum training-loss checkpoint; no heldout data used for selection','comparison_scope':'supplemental sixfold evaluation, not replacement of five-seed main experiment','packaged_core_sha256':hashlib.sha256(V45.read_bytes()).hexdigest()},indent=2))
def raw_cache(m,path):
    if path.exists() and path.with_suffix('.complete').exists(): return np.load(path,mmap_mode='r')
    shape=(int(m.window_count.sum()),2048,3); x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape); offset=0
    for bearing,g in m.groupby('bearing_code',sort=False):
        print('EXTRACT',bearing,flush=True)
        with tempfile.TemporaryDirectory(prefix='raw_extract_',dir=OUT) as tmp:
            subprocess.run(['unrar','x','-o+','-inul',str(RAW/f'{bearing}.rar'),tmp+'/'],check=True)
            files={p.name:p for p in Path(tmp).rglob('*.mat')}
            for r in g.itertuples():
                obj=loadmat(files[r.source_mat],simplify_cells=True,variable_names=[Path(r.source_mat).stem])[Path(r.source_mat).stem]
                channels={z['Name']:np.asarray(z['Data'],float) for z in obj['Y']}; a=np.column_stack([channels[k] for k in ['phase_current_1','phase_current_2','vibration_1']]); count=int(r.window_count); a=a[:count*2048].reshape(count,2048,3)
                assert np.isfinite(a).all(); x[offset:offset+count]=a; offset+=count
    assert offset==shape[0]; x.flush(); path.with_suffix('.complete').write_text('raw float32; no cross-bearing normalization'); return np.load(path,mmap_mode='r')
xn=raw_cache(normal,OUT/'normal_raw_f32.npy'); cn=nm.condition.to_numpy(); gn=nm.file_id.to_numpy(); bn=nm.bearing.to_numpy(); prepared=[]
for f in folds:
    d=OUT/f'fold_{f["fold"]}'; d.mkdir(exist_ok=True); fit=np.flatnonzero(np.isin(bn,f['fit_normal'])); cal=np.flatnonzero(bn==f['cal_normal']); test=np.flatnonzero(bn==f['test_normal'])
    assert not np.intersect1d(fit,cal).size and not np.intersect1d(fit,test).size and not np.intersect1d(cal,test).size
    assert set(cn[fit])==set(cn[cal])==set(cn[test])=={0,1,2,3}
    train={'fit_x':xn,'fit_cond':cn,'fit_group':gn,'fit_ids':fit}
    mean,std=v.normalization(xn,cn,fit); np.savez(d/'fit_only_normalization.npz',mean=mean,std=std)
    if (d/'TRAIN_CAL_FROZEN.json').exists():
        prepared.append((f,d,fit,cal,test)); continue
    print('TRAIN_FOLD',f,flush=True)
    model=v.fit_encoder('supcon_condition',train,mean,std,SEED,d)
    zfit=v.embed(model,xn,cn,fit,mean,std); zcal=v.embed(model,xn,cn,cal,mean,std); calraw=np.empty(len(cal),np.float32); banks={}
    for cc in sorted(np.unique(cn[fit])):
        models,priors,counts=v.fit_gmm(zfit[cn[fit]==cc],SEED+100*int(cc)); banks[int(cc)]=(models,priors); calraw[cn[cal]==cc]=v.gmm_nll(zcal[cn[cal]==cc],models,priors)
    calscore,_,pars=v.calibrate(calraw,np.empty(0,np.float32),cn[cal],np.empty(0,int),'condition')
    threshold=float(np.quantile(calscore,.95)); joblib.dump(banks,d/'fit_only_gaussian_banks.joblib'); np.save(d/'calibration_score.npy',calscore)
    (d/'TRAIN_CAL_FROZEN.json').write_text(json.dumps({**f,'seed':SEED,'fit_windows':len(fit),'cal_windows':len(cal),'normal_test_windows':len(test),'calibration':pars,'threshold':threshold,'test_features_read_before_checkpoint_and_bank_frozen':False,'fault_raw_opened_before_all_training_frozen':False},indent=2))
    prepared.append((f,d,fit,cal,test)); del model,zfit,zcal; torch.cuda.empty_cache()
(OUT/'ALL_SIX_MODELS_FROZEN_BEFORE_FAULT_OPEN').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
xf=raw_cache(faulty,OUT/'fault_raw_f32.npy'); cf=fm.condition.to_numpy(); bf=fm.bearing.to_numpy(); rows=[]; cell=[]
for f,d,fit,cal,test in prepared:
    if (d/'metrics.json').exists(): rows.append(json.loads((d/'metrics.json').read_text())); continue
    norm=np.load(d/'fit_only_normalization.npz'); mean,std=norm['mean'],norm['std']; frozen=json.loads((d/'TRAIN_CAL_FROZEN.json').read_text()); banks=joblib.load(d/'fit_only_gaussian_banks.joblib')
    model=v.PatchEncoder(in_channels=3,projection_dim=256,use_revin=False).to(v.DEVICE); model.load_state_dict(torch.load(d/'model.pt',map_location=v.DEVICE,weights_only=True))
    zn=v.embed(model,xn,cn,test,mean,std); zf=v.embed(model,xf,cf,np.arange(len(xf)),mean,std); nr=np.empty(len(test),np.float32); fr=np.empty(len(xf),np.float32)
    for cc,(models,priors) in banks.items():
        nr[cn[test]==cc]=v.gmm_nll(zn[cn[test]==cc],models,priors); fr[cf==cc]=v.gmm_nll(zf[cf==cc],models,priors)
    sn=np.empty_like(nr); sf=np.empty_like(fr)
    for cc,p in frozen['calibration'].items():
        cc=int(cc); sn[cn[test]==cc]=(nr[cn[test]==cc]-p['median'])/p['iqr']; sf[cf==cc]=(fr[cf==cc]-p['median'])/p['iqr']
    np.save(d/'normal_test_score.npy',sn); np.save(d/'fault_test_score.npy',sf)
    y=np.r_[np.zeros(len(sn),int),np.ones(len(sf),int)]; scores=np.r_[sn,sf]; pred=scores>=frozen['threshold']; tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
    row={**f,'seed':SEED,'AUROC':float(roc_auc_score(y,scores)),'F1':float(f1_score(y,pred)),'normal_FPR':float(fp/(tn+fp)),'fault_TPR':float(tp/(tp+fn)),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),'threshold':frozen['threshold']}; rows.append(row); (d/'metrics.json').write_text(json.dumps(row,indent=2)); pd.DataFrame(rows).to_csv(OUT/'sixfold_metrics.csv',index=False)
    print('FOLD_RESULT',json.dumps(row),flush=True); del model,zn,zf; torch.cuda.empty_cache()
# Complete-bearing two-factor resampling, shared damaged cohort never counted 6x.
rng=np.random.default_rng(20261013); B=5000; wh=np.stack([np.bincount(rng.integers(0,6,6),minlength=6) for _ in range(B)]); wf=np.stack([np.bincount(rng.integers(0,14,14),minlength=14) for _ in range(B)]); aucdraw=[]; fdraw=[]
for f,d,fit,cal,test in prepared:
    sn=np.load(d/'normal_test_score.npy'); sf=np.load(d/'fault_test_score.npy'); th=json.loads((d/'TRAIN_CAL_FROZEN.json').read_text())['threshold']; ss=np.sort(sn); nn=len(sn); fp=int((sn>=th).sum()); counts=[]; sums=[]; tp=[]; fn=[]
    for code in faultcodes:
        z=sf[bf==code]; counts.append(len(z)); sums.append(float((np.searchsorted(ss,z,'left')+.5*(np.searchsorted(ss,z,'right')-np.searchsorted(ss,z,'left'))).sum())); pp=z>=th; tp.append(int(pp.sum())); fn.append(int((~pp).sum()))
    counts=np.asarray(counts); sums=np.asarray(sums); tp=np.asarray(tp); fn=np.asarray(fn)
    aucdraw.append((wf@sums)/(nn*(wf@counts))); t=wf@tp; n=wf@fn; fdraw.append(2*t/(2*t+n+fp))
    cell.append({'fold':f['fold'],'test_normal':f['test_normal'],'normal_n':nn,'FP':fp,'fault_clusters':[{'bearing':c,'n':int(counts[j]),'TP':int(tp[j]),'FN':int(fn[j]),'pairwise_rank_sum':float(sums[j])} for j,c in enumerate(faultcodes)]})
ad=np.stack(aucdraw,1); fd=np.stack(fdraw,1); ab=(ad*wh).sum(1)/6; fb=(fd*wh).sum(1)/6
summary={'mean_fold_AUROC':float(np.mean([r['AUROC'] for r in rows])),'mean_fold_F1':float(np.mean([r['F1'] for r in rows])),'mean_normal_FPR':float(np.mean([r['normal_FPR'] for r in rows])),'mean_fault_TPR':float(np.mean([r['fault_TPR'] for r in rows])),'conditional_cluster_CI_AUROC':np.quantile(ab,[.025,.975]).tolist(),'conditional_cluster_CI_F1':np.quantile(fb,[.025,.975]).tolist(),'bootstrap_draws':B,'bootstrap_scope':'stratified physical healthy(6) and damaged(14) bearing clusters; normal identity carries its held-out fold; identical damaged multiplicities used across all six folds; frozen fitted pipelines, overlapping fitting sets; does not include retraining uncertainty or new campaigns','seed_replicates':1,'normal_physical_test_bearings':6,'fault_physical_test_bearings':14}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)); (OUT/'bootstrap_cluster_statistics.json').write_text(json.dumps(cell,indent=2)); pd.DataFrame({'draw':np.arange(B),'macro_AUROC':ab,'macro_F1':fb}).to_csv(OUT/'bootstrap_draws.csv',index=False)
shutil.copy2(__file__,OUT/'paderborn_sixfold.py'); shutil.copy2(V45,OUT/'source_runner_v45.py'); (OUT/'COMPLETE').write_text('six folds complete; no main-result changes')
print('COMPLETE',json.dumps(summary),flush=True)
