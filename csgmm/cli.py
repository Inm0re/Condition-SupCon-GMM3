import argparse,json
from pathlib import Path
import numpy as np
from . import pipeline

def main():
    p=argparse.ArgumentParser(description='Train/evaluate the public normal-only detector')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True,help='User-generated preprocessed array directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['train','evaluate','all'],default='all')
    p.add_argument('--variant',choices=['H','A0','A1','A2','A4','N1','N2','N3','N4'],default='H')
    p.add_argument('--seed',type=int,nargs='+',default=None)
    p.add_argument('--device',choices=['cpu','cuda'],default=None)
    p.add_argument('--steps',type=int,default=None,help='Override for a documented sensitivity or smoke run')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.steps is not None:cfg['steps']=a.steps
    for seed in a.seed or cfg['seeds']:
        out=a.output/f'seed_{seed}'/a.variant
        if a.stage in ['train','all']:pipeline.fit(a.data,out,cfg,seed,a.variant,a.device)
        if a.stage in ['evaluate','all']:
            print(json.dumps(pipeline.evaluate(a.data,out,cfg['dataset'],a.device),indent=2))

if __name__=='__main__':main()
