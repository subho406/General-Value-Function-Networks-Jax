import sys
import argparse
import pandas as pd
from numpy.core.fromnumeric import argsort
import tqdm
import os
import numpy as np
import wandb
sys.path.append('../')
sys.path.append('.')

from src.utils.compassworld.predictors import *
from src.envs import CompassWorld
from src.utils.compassworld.policies import CompassWorldRandomPolicy


def run_compass_world(args):
    env=CompassWorld(args.env_size,args.env_size,seed=args.seed)
    policy=CompassWorldRandomPolicy(seed=args.seed)
    o_tplus1=env.observe()
    o_t=None
    a_t=None
    o_tminus1=None
    a_tminus1=None
    obs_size=env.color_vec_size()
    act_size=env.action_vec_size()
    if args.agent_type=='prnn':
        predictor=CWPRNNPredictor(obs_size,act_size,args.truncation,
                                                rnn_hidden_size=args.rnn_hidden_size,hidden_size=args.hidden_size,
                                                lr=args.lr,optimizer=args.optimizer,seed=args.seed)
    elif args.agent_type=='gvfn':
        predictor=GVFNTDPredictor(obs_size,act_size,args.truncation,
                                    rnn_hidden_size=args.rnn_hidden_size,hidden_size=args.hidden_size,
                                                lr=args.lr,optimizer=args.optimizer,seed=args.seed)
    losses,rmsves,accs=[],[],[]
    mean_accs,mean_rmsves,steps=[],[],[]

    for i in tqdm.tqdm(range(args.steps)):
        #update the current observations and actions 
        o_tminus1=o_t
        a_tminus1=a_t
        o_t=o_tplus1
        target_t=env.wall_ahead()
        a_t,mu_otat=policy.step(o_t)
        o_tplus1=env.step(a_t)
        #Vectorize action, observation
        
        if i>0:
            #Take a step with the predictor
            loss,pred=predictor.step(o_t,a_tminus1,o_tplus1,a_t, mu_otat)
            losses.append(loss)
            error=pred-predictor.vectorize_target(target_t)
            rmsve=((error**2).mean())**0.5
            rmsves.append(rmsve)
            acc=float(pred.argmax()==predictor.colors_to_idx[target_t])
            accs.append(acc)
            if (i+1)%args.eval_steps==0:
                mean_loss=np.mean(losses)
                mean_rmsve=np.mean(rmsves)
                mean_acc=np.mean(accs)
                mean_accs.append(mean_acc)
                mean_rmsves.append(mean_rmsve)
                steps.append(i+1)
                tqdm.tqdm.write('Step: %d, Accuracy: %f, RMSVE: %f, Loss: %f'%(i+1,mean_acc,mean_rmsve,mean_loss))
                if args.wandb: wandb.log({'accuracy':mean_acc,'rmsve':mean_rmsve,'loss':mean_loss},step=i+1)
                rmsves=[]
                accs=[]
                losses=[]
    print("Training Completed...")
    if args.output_dir is not None:
        df=pd.DataFrame({'steps':steps,'rmsve':mean_rmsves,'acc':mean_accs})
        df.to_csv(os.path.join(args.output_dir,'compassworld_%s_%d_%d_%d.csv'%(args.agent_type,args.truncation,
                                args.steps,args.seed)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--truncation',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--rnn_hidden_size',type=int,default=40)
    parser.add_argument('--env_size',type=int,default=6)
    parser.add_argument('--hidden_size',type=int,default=32)
    parser.add_argument('--steps',type=int,default=1000000)
    parser.add_argument('--eval_steps',type=int,default=100)
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument("--wandb", default=False, action="store_true",
                    help="Use wandb for logging.")
    parser.add_argument('--agent_type',type=str,default='prnn',help='Agent to use for training. Available options: [prnn, gvfn]')
    parser.add_argument('--output_dir',type=str,default=None,help='Dump the output to numpy array')
    parser.add_argument('--optimizer',type=str,default='sgd',help='Optimizer to use. One of [sgd, adam]')
    args=parser.parse_args()
    if args.wandb: wandb.init(project='GVFN',config=args)
    run_compass_world(args)
