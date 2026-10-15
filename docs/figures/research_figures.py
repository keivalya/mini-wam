"""Figures from simulator frames and archived numeric results; no generated imagery."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from envs.toy_point_env import ToyPointEnv
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

env=ToyPointEnv(seed=71);env.reset(seed=71,target='blue')
frames=[]
for t in range(25):
 if t in [0,6,12,24]:frames.append((t,env.render().copy()))
 env.step(env.expert_action(noise=0))
fig,axs=plt.subplots(1,4,figsize=(6.3,1.8))
for ax,(t,im) in zip(axs,frames):
 ax.imshow(im,interpolation='nearest');ax.set_title(f't = {t}');ax.axis('off')
fig.suptitle('Simulator observations: go to the blue goal',y=1.04)
save(fig,'research-environment')

d=np.load(ROOT/'docs/results/figure-data/episode_strip.npz')
fig,axs=plt.subplots(3,5,figsize=(6.3,3.8))
for r,key in enumerate(['real','recon','imagined']):
 for j,k in enumerate([0,1,3,5,7]):
  axs[r,j].imshow(d[key][k],interpolation='nearest');axs[r,j].set_xticks([]);axs[r,j].set_yticks([])
  if r==0:axs[r,j].set_title(f't = {d["timesteps"][k]}')
  if j==0:axs[r,j].set_ylabel({'real':'Observed','recon':'Reconstructed','imagined':'Predicted'}[key])
fig.subplots_adjust(wspace=.06,hspace=.08)
save(fig,'research-observations')

fig,axs=plt.subplots(1,2,figsize=(6.3,2.5))
for name,color,label in [('explore-0.0','#0072B2','Expert only'),('baseline','#D55E00','Burst probability 0.3'),('explore-0.6','#009E73','Burst probability 0.6')]:
 es=[json.loads(p.read_text()) for p in sorted((ROOT/'docs/results'/name).glob('seed*/wam_eval.json'))]
 for ax,key in zip(axs,['state_nmse','cf']):
  vals=np.array([e[key][1:] if key!='cf' else e['counterfactual']['wam_nmse'] for e in es]);x=np.arange(1,vals.shape[1]+1)
  ax.plot(x,vals.mean(0),label=label,color=color);ax.fill_between(x,vals.mean(0)-vals.std(0),vals.mean(0)+vals.std(0),color=color,alpha=.15)
  ax.set_yscale('log');ax.set_xlabel('Prediction horizon');ax.set_ylabel('Normalized MSE')
axs[0].set_title('Factual: six state dimensions');axs[1].set_title('Counterfactual: four dimensions');axs[0].legend(fontsize=7)
fig.tight_layout();save(fig,'research-coverage')
