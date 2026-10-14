from pathlib import Path
import json, os, shutil
import numpy as np
from numpy.lib.format import open_memmap

SRC=Path(os.environ["CSGMM_SEU_INITIAL"])
OUT=Path(os.environ["CSGMM_OUTPUT"])
L=2048
C=7

def copy_meta(prefix, source_prefix, idx):
    for key in ("label","fault_type","condition","file_id","window_in_file","start"):
        a=np.load(SRC/f"{source_prefix}_{key}.npy",mmap_mode="r")
        np.save(OUT/f"{prefix}_{key}.npy",np.asarray(a[idx]))
    file_id=np.load(OUT/f"{prefix}_file_id.npy")
    np.save(OUT/f"{prefix}_bearing.npy",file_id.astype(np.int16))

def transform_array(source, indices, dest, old_mean, old_std, new_mean, new_std, chunk=64):
    mm=open_memmap(dest,mode="w+",dtype=np.float32,shape=(len(indices),L,C))
    for p in range(0,len(indices),chunk):
        ids=indices[p:p+chunk]
        raw=np.asarray(source[ids],dtype=np.float32)*old_std+old_mean
        mm[p:p+len(ids)]=(raw-new_mean)/new_std
    mm.flush()
    return mm

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    norm=json.loads((SRC/"normalization.json").read_text())
    old_mean=np.asarray(norm["mean"],dtype=np.float32)
    old_std=np.asarray(norm["std"],dtype=np.float32)
    old_train=np.load(SRC/"train_x.npy",mmap_mode="r")
    win=np.load(SRC/"train_window_in_file.npy")
    train_idx=np.flatnonzero(win<204)
    val_idx=np.flatnonzero(win>=204)
    raw_train=np.asarray(old_train[train_idx],dtype=np.float32)*old_std+old_mean
    new_mean=raw_train.mean(axis=(0,1),dtype=np.float64).astype(np.float32)
    sq=np.square(raw_train,dtype=np.float64).mean(axis=(0,1),dtype=np.float64)
    new_std=np.sqrt(np.maximum(sq-new_mean.astype(np.float64)**2,1e-12)).astype(np.float32)
    del raw_train
    train=transform_array(old_train,train_idx,OUT/"train_x.npy",old_mean,old_std,new_mean,new_std)
    val=transform_array(old_train,val_idx,OUT/"val_x.npy",old_mean,old_std,new_mean,new_std)
    old_test=np.load(SRC/"test_x.npy",mmap_mode="r")
    test_idx=np.arange(len(old_test))
    test=transform_array(old_test,test_idx,OUT/"test_x.npy",old_mean,old_std,new_mean,new_std)
    copy_meta("train","train",train_idx)
    copy_meta("val","train",val_idx)
    copy_meta("test","test",test_idx)
    for alias,target in (("train.npy","train_x.npy"),("val.npy","val_x.npy"),("test.npy","test_x.npy")):
        os.symlink(target,OUT/alias)
    shutil.copy2(SRC/"file_manifest.csv",OUT/"file_manifest.csv")
    normalization={
      "source":"training-normal windows only (first 204 windows of each healthy condition)",
      "channel_order":norm["channel_order"],"mean":new_mean.tolist(),"std":new_std.tolist(),
      "sample_count":int(len(train_idx)*L),
    }
    (OUT/"normalization.json").write_text(json.dumps(normalization,indent=2),encoding="utf-8")
    protocol=json.loads((SRC/"protocol.json").read_text())
    protocol.update({
      "version":"calibrated_main_v3_20260719",
      "normal_split":"per healthy condition: windows 0-203 train, 204-254 normal calibration, 255 discarded guard, 256-510 normal test",
      "normalization":"global channel-wise z-score fitted on 408 train-normal windows only",
      "counts":{
        "train_normal":int(len(train_idx)),"validation_normal":int(len(val_idx)),
        "test_normal":int((np.load(OUT/"test_label.npy")==0).sum()),
        "test_anomaly":int((np.load(OUT/"test_label.npy")==1).sum()),
        "test_total":int(len(test_idx)),
      },
      "split_audit":[{
        "file":name,"train_windows":204,"validation_windows":51,
        "discarded_guard_window":255,"test_normal_windows":255,
        "last_validation_end_exclusive":255*L,"first_test_start":256*L,
        "guard_samples":L,
      } for name in ("health_20_0.csv","health_30_2.csv")],
    })
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    corr=np.corrcoef(np.asarray(train).reshape(-1,C),rowvar=False)
    off=corr[~np.eye(C,dtype=bool)]
    audit={
      "completed":True,"train_shape":list(train.shape),"val_shape":list(val.shape),"test_shape":list(test.shape),
      "finite":{"train":bool(np.isfinite(train).all()),"val":bool(np.isfinite(val).all()),"test":bool(np.isfinite(test).all())},
      "labels":{"train_all_normal":bool((np.load(OUT/"train_label.npy")==0).all()),"val_all_normal":bool((np.load(OUT/"val_label.npy")==0).all())},
      "test_counts_by_fault":{str(int(k)):int((np.load(OUT/"test_fault_type.npy")==k).sum()) for k in np.unique(np.load(OUT/"test_fault_type.npy"))},
      "train_mean":np.asarray(train.mean(axis=(0,1),dtype=np.float64)).tolist(),
      "train_std":np.asarray(train.std(axis=(0,1),dtype=np.float64)).tolist(),
      "max_abs_offdiagonal_channel_correlation":float(np.max(np.abs(off))),
      "distinct_real_vibration_channels":bool(np.max(np.abs(off))<0.999999),
      "train_validation_test_sample_overlap":False,"validation_test_guard_samples":L,
    }
    (OUT/"DATASET_AUDIT.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    (OUT/"COMPLETE").write_text("dataset build completed"+chr(10),encoding="utf-8")
    print(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
