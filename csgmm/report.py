import argparse,json,csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for f in sorted(a.runs.rglob('metrics.json')):
        protocol=f.parent/'fit_protocol.json'
        if not protocol.exists():continue
        r=json.loads(f.read_text());m=json.loads(protocol.read_text())
        rows.append(dict(dataset=m['dataset'],seed=m['seed'],variant=m['variant'],**r))
    if not rows:raise ValueError('No public run metrics found')
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (a.output/'per_seed_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    metrics=['auroc','auprc','f1','mcc','normal_fpr','tpr','specificity','balanced_accuracy'];summary=[]
    for dataset,variant in sorted(set((r['dataset'],r['variant']) for r in rows)):
        rr=[r for r in rows if r['dataset']==dataset and r['variant']==variant]
        out=dict(dataset=dataset,variant=variant,n_seeds=len(rr))
        for k in metrics:
            values=[r[k] for r in rr];out[k+'_mean']=float(np.mean(values));out[k+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(out)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[r['dataset']+'/'+r['variant'] for r in summary]
    fig,axes=plt.subplots(1,3,figsize=(12,max(3,len(labels)*.3)),constrained_layout=True)
    for ax,k in zip(axes,['auroc','f1','normal_fpr']):
        ax.errorbar([r[k+'_mean'] for r in summary],np.arange(len(labels)),xerr=[r[k+'_sd'] or 0 for r in summary],fmt='o')
        ax.set_yticks(np.arange(len(labels)),labels);ax.set_xlabel(k);ax.grid(axis='x',alpha=.2)
    fig.savefig(a.output/'public_results.png',dpi=300);fig.savefig(a.output/'public_results.pdf')

if __name__=='__main__':main()
