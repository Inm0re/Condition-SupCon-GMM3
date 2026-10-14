import argparse
from pathlib import Path
import numpy as np
from .pipeline import predict

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--windows',type=Path,required=True,help='NL C-format preprocessed .npy inputs')
    p.add_argument('--conditions',type=Path,required=True)
    p.add_argument('--artifact',type=Path,required=True,help='Locally trained artifact directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'])
    a=p.parse_args()
    score,pred,threshold=predict(np.load(a.windows,mmap_mode='r',allow_pickle=False),np.load(a.conditions,allow_pickle=False),a.artifact,a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.output,score=score,prediction=pred,threshold=threshold)

if __name__=='__main__':main()
