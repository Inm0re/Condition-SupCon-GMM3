from pathlib import Path
import json
import numpy as np

def build(raw,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    buckets={s:[] for s in ['train','val','test']}
    # Existing operational labels; not inferred physical degradation onsets.
    for condition,(name,normal_n,total) in enumerate([('Bearing1_1',78,123),('Bearing2_2',46,161)]):
        paths=[p for p in Path(raw).rglob(name) if p.is_dir()]
        if len(paths)!=1:raise ValueError(f'Expected one directory named {name}')
        files=sorted(paths[0].glob('*.csv'),key=lambda p:int(p.stem))
        if len(files)!=total:raise ValueError(f'{name}: expected {total} records')
        fit_end=int(np.floor(.6*normal_n));cal_end=int(np.floor(.8*normal_n))
        for index,file in enumerate(files):
            value=np.loadtxt(file,delimiter=',',skiprows=1,dtype=np.float32)
            if value.shape!=(32768,2) or not np.isfinite(value).all():raise ValueError(f'Unexpected record {file}')
            # Retain the original 16-window float16 corpus interface. Archive
            # numerical parity is checked separately; fresh training is required.
            windows=value.reshape(16,2048,2).astype(np.float16)
            split='train' if index<fit_end else ('val' if index<cal_end else 'test')
            buckets[split].append((windows,condition,index+1,int(index>=normal_n)))
    for split,rows in buckets.items():
        np.save(output/f'{split}_x.npy',np.concatenate([r[0] for r in rows]))
        for key,col in [('condition',1),('record',2),('label',3)]:
            np.save(output/f'{split}_{key}.npy',np.concatenate([np.full(16,r[col],np.int32) for r in rows]))
    (output/'protocol.json').write_text(json.dumps(dict(bearings=['Bearing1_1','Bearing2_2'],normal_records=[78,46],fault_onset_1based=[79,47],windows_per_record=16,window_length=2048,input_channels=['horizontal','vertical'],record_ids_disjoint=True,same_bearings_across_partitions=True),indent=2))
