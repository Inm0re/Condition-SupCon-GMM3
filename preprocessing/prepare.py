"""Rebuild public arrays from user-owned raw inputs; never download or publish data."""
import argparse,os,runpy,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['hust','paderborn_real','seu_dds','xjtu_sy'],required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.raw.resolve();out=a.output.resolve()
    if not raw.is_dir():raise FileNotFoundError(raw)
    if out.exists():raise FileExistsError('Choose a new output directory; existing data are not overwritten')
    if out==raw or raw.is_relative_to(out):raise ValueError('Output must not contain the raw-input directory')
    if (out.parent/(out.name+'.extract-tmp')).exists():raise FileExistsError('Paderborn extraction staging already exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    os.environ.update(CSGMM_RAW=str(raw),CSGMM_OUTPUT=str(out))
    base=Path(__file__).parent
    if a.dataset=='hust':
        sys.argv=['hust_source.py','--source',str(raw),'--output',str(out)]
        runpy.run_path(str(base/'hust_source.py'),run_name='__main__')
        print('HUST training array directory:',out/'windows_L2048_H256')
    elif a.dataset=='paderborn_real':runpy.run_path(str(base/'paderborn_source.py'),run_name='__main__')
    elif a.dataset=='seu_dds':
        initial=out.parent/(out.name+'.initial')
        if initial.exists():raise FileExistsError(initial)
        os.environ['CSGMM_OUTPUT']=str(initial)
        runpy.run_path(str(base/'seu_initial_source.py'),run_name='__main__')
        os.environ.update(CSGMM_OUTPUT=str(out),CSGMM_SEU_INITIAL=str(initial))
        runpy.run_path(str(base/'seu_calibration_source.py'),run_name='__main__')
    else:
        from xjtu_source import build
        build(raw,out)

if __name__=='__main__':main()
