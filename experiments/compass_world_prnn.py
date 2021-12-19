import sys
import argparse
from numpy.core.fromnumeric import argsort
import tqdm
import numpy as np
import optax
sys.path.append('../')
sys.path.append('.')

from src.predictors.compassworld_wall import CompassWorldPRNNWallPredictor
from src.envs import CompassWorld
from src.policies import CompassWorldRandomPolicy

def run_compass_world(truncation,rnn_hidden_size,hidden_size,steps,lr,seed):
    env=CompassWorld(seed=seed)
    policy=CompassWorldRandomPolicy(seed=seed)
    obs=env.observe()
    obs_size=env.color_vec_size()
    act_size=env.action_vec_size()
    predictor=CompassWorldPRNNWallPredictor(obs_size,act_size,truncation,
                                            rnn_hidden_size=rnn_hidden_size,hidden_size=hidden_size,
                                           lr=lr,seed=seed)
    losses=[]
    for i in tqdm.tqdm(range(steps)):
        action=policy.step(obs)
        new_obs=env.step(action)
        target_vec=env.wall_ahead()
        #Vectorize action, observation
        new_obs_vec=env.vectorize_color(new_obs)
        action_vec=env.vectorize_action(action)
        
        #Take a step with the predictor
        loss,pred=predictor.step(new_obs_vec,action_vec,target_vec)
        obs=new_obs
        losses.append(loss)
        if i%10==0:
            print(np.mean(losses))
            losses=[]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--truncation',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--rnn_hidden_size',type=int,default=40)
    parser.add_argument('--hidden_size',type=int,default=32)
    parser.add_argument('--steps',type=int,default=1000000)
    parser.add_argument('--seed',type=int, default=0)
    args=parser.parse_args()
    run_compass_world(truncation=args.truncation,rnn_hidden_size=args.rnn_hidden_size,
                        hidden_size=args.hidden_size,steps=args.steps,lr=args.lr,
                        seed=args.seed)
