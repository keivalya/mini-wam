"""Matched pixel-input pilot: BC vs current-state vs current+future supervision.
Run collection separately; all conditions share H=5 starts, encoder initialization,
expert mask and minibatch order. No pixel/latent loss: isolates state targets.
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.train import TrainingDataset, split_episodes
from models.wam_policy import MiniWAMPolicy
from envs.toy_point_env import ToyPointEnv
from utils.tokenizer import SimpleTokenizer

@torch.no_grad()
def test(model,vocab,seed,n=100):
    envs=[ToyPointEnv(seed=seed+i) for i in range(n)]
    obs=[e.reset(seed=seed+i) for i,e in enumerate(envs)]
    tok=SimpleTokenizer(vocab=vocab)
    tokens=torch.tensor([tok.encode(o[2]['instruction']) for o in obs],device=model.state_mean.device)
    success=np.zeros(n,dtype=bool);lengths=np.full(n,80,dtype=int)
    for t in range(80):
        im=torch.tensor(np.stack([o[0] for o in obs]),device=tokens.device).permute(0,3,1,2).float()/255
        states=torch.tensor(np.stack([o[1] for o in obs]),device=tokens.device)
        actions=model.act(im,tokens,states).cpu().numpy()
        for i,e in enumerate(envs):
            if not success[i]:
                image,state,_,done,info=e.step(actions[i]);obs[i]=(image,state,info)
                if done:success[i]=True;lengths[i]=t+1
        if success.all():break
    return {'success_rate':float(success.mean()),'success':success.tolist(),'lengths':lengths.tolist(),'episode_seeds':list(range(seed,seed+n))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=15);p.add_argument('--device',default='cuda')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    data=TrainingDataset(args.dataset,horizon=5)
    train,val=split_episodes(data.episode_ids,.1,args.seed)
    starts=data.starts_in(train);mask=np.isin(data.episode_ids,train);data.to(args.device)
    manifest={'args':vars(args),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
      'dataset_sha256':hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
      'source_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for pattern in ['models/*.py','scripts/supervision_ablation.py','scripts/train.py','envs/toy_point_env.py'] for q in Path('.').glob(pattern)},
      'train_episodes':train.tolist(),'validation_episodes':val.tolist(),'training_starts':len(starts),
      'protocol':'Exploratory pilot, not replication of archived full WAM. Equal H5 starts, same seed and ordering; current+future state prediction only.',
      'device':torch.cuda.get_device_name() if args.device=='cuda' else args.device}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    results={}
    for condition in ['bc','current','future']:
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        # Same full initialization ensures encoder/head weights match exactly.
        model=MiniWAMPolicy(vocab_size=max(data.vocab.values())+1,state_dim=6,action_dim=2,
          use_state_input=False,predict_state=True,predict_image=False).to(args.device)
        model.set_state_stats(data.states[mask].mean(0),data.states[mask].std(0))
        optim=torch.optim.Adam(model.parameters(),lr=1e-3);history=[];begin=time.time()
        for epoch in range(args.epochs):
            model.train();torch.manual_seed(args.seed*10000+epoch);total=0.;count=0
            for b in data.batches(starts,64,shuffle=True):
                z=model.encode_obs(b['image'],b['text_ids'],b['state']);expert=b['is_expert']>.5
                loss=model.policy_head.loss(b['action'][expert],z[expert]) if expert.any() else z.sum()*0
                if condition!='bc':loss=loss+.5*F.mse_loss(model.state_readout(z),model.normalize_state(b['state']))
                if condition=='future':
                    zp=model.dynamics.rollout(z,b['actions'])
                    loss=loss+.5*F.mse_loss(model.state_readout(zp),model.normalize_state(b['future_states']))
                optim.zero_grad();loss.backward();optim.step();total+=loss.item()*len(z);count+=len(z)
            history.append(total/count)
            print(condition,epoch+1,history[-1],flush=True)
        model.eval();results[condition]=test(model,data.vocab,50000+1000*args.seed)
        results[condition].update(training_seconds=time.time()-begin,loss_history=history,updates=args.epochs*((len(starts)+63)//64))
        torch.save({'model_state_dict':model.state_dict(),'model_config':model.config,'vocab':data.vocab,'condition':condition,'manifest':manifest},out/(condition+'.pt'))
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print('RESULT',condition,results[condition]['success_rate'],flush=True)
if __name__=='__main__':main()
