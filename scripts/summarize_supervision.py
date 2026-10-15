"""Validate and summarize the complete matched-supervision pilot (not a significance test)."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',type=int,default=5);a=p.parse_args()
 paths=sorted(a.root.glob('seed-*/ablation/results.json'))
 if len(paths)!=a.seeds:raise ValueError(f'Expected {a.seeds} completed seeds; found {len(paths)}')
 rows=[];hashes=[]
 for path in paths:
  d=json.loads(path.read_text());m=json.loads((path.parent/'manifest.json').read_text())
  assert set(d)=={'bc','current','future'},path
  assert len({v['updates'] for v in d.values()})==1,path
  assert all(v['episode_seeds']==d['bc']['episode_seeds'] for v in d.values()),path
  for v in d.values():
   assert len(v['success'])==len(v['lengths'])==len(v['episode_seeds'])==100
   assert abs(np.mean(v['success'])-v['success_rate'])<1e-12
  rows.append({'seed':m['args']['seed'],**{k:d[k]['success_rate'] for k in d}});hashes.append(m['source_sha256'])
 assert len({r['seed'] for r in rows})==a.seeds
 assert all(h==hashes[0] for h in hashes),'Source differs between jobs'
 result={'description':'Exploratory pilot; no significance or convergence claim. Success means/sample SD over training seeds.',
         'n_seeds':a.seeds,'rows':rows,'summary':{}}
 for key in ['bc','current','future']:
  x=np.array([r[key] for r in rows]);result['summary'][key]={'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1))}
 for lhs,rhs in [('current','bc'),('future','current'),('future','bc')]:
  x=np.array([r[lhs]-r[rhs] for r in rows]);result['summary'][lhs+'-minus-'+rhs]={'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),'values':x.tolist()}
 (a.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
