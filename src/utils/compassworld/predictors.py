import jax 
import abc
import haiku as hk
from jax._src.random import gamma
import optax
import jax.numpy as jnp
import time

from src.models import MultiplicativeRNN
from src.agent import GVF, Horde,initialize_gvf
from src.envs import CompassWorld
from jax import random,jit, value_and_grad,lax
from functools import partial


class TerminatingHorizonGVF(GVF):
    def __init__(self,color,gamma) -> None:
        super().__init__()
        self.color=color
        self.gamma=gamma
        self.f_vec=CompassWorld.vectorize_action('f')
        self.w_vec=CompassWorld.vectorize_color('w')
        self.color_vec=CompassWorld.vectorize_color(self.color)
        def check_vec_equal(vec1,vec2):
            return (vec1==vec2).sum()==vec1.shape[0]
        self.check_vec_equal=check_vec_equal
    
    def __call__(self, obs, act, prev_state):
        cumulant=lax.cond(self.check_vec_equal(obs,self.color_vec),lambda x: jnp.array(1.0),lambda x: jnp.array(0.0),None)
        gamma=lax.cond(self.check_vec_equal(obs,self.w_vec),lambda x: jnp.array(x),lambda x: jnp.array(0.0),self.gamma)
        policy=lax.cond(self.check_vec_equal(act,self.f_vec),lambda x: jnp.array(1.0),lambda x: jnp.array(0.0),None)
        return policy,cumulant,gamma,None

    @staticmethod
    def initial_state():
        return None


class CompassWorldPredictor:
    def __init__(self,obs_size,act_size,truncation,rnn_activation_type='sigmoid',
                rnn_hidden_size=40,hidden_size=32,lr=0.001,optimizer='sgd',
                seed=0) -> None:
        self.key=random.PRNGKey(seed)
        self.rnn_hidden_size=rnn_hidden_size
        self.hidden_size=hidden_size
        self.obs_size=obs_size
        self.act_size=act_size
        self.truncation=truncation
        #Initialize the RNN layer
        def rnn_forward(obs,last_act,last_state):
            rnn=MultiplicativeRNN(self.obs_size,self.act_size,self.rnn_hidden_size,
                                activation=rnn_activation_type)
            out,state=rnn(obs,last_act,last_state)
            return out, state
        self.last_state=MultiplicativeRNN.initial_state(self.obs_size,self.act_size,self.rnn_hidden_size,
                                                        self.truncation)
        rnn_forward_trf=hk.without_apply_rng(hk.transform(rnn_forward))
        self.sample_o=jax.random.normal(self.key,[self.obs_size])
        self.sample_a=jax.random.normal(self.key,[self.act_size])
        self.rnn_params=rnn_forward_trf.init(self.key,self.sample_o,self.sample_a,self.last_state)
        self.rnn_forward=jit(rnn_forward_trf.apply)
        #Initilize the output layers
        def output_forward(hidden_state):
            output_layer=hk.Sequential([hk.Linear(self.hidden_size),jax.nn.relu,
                            hk.Linear(5)])
            pred=output_layer(hidden_state)
            return pred
        output_forward_trf=hk.without_apply_rng(hk.transform(output_forward))
        sample_h=jax.random.normal(self.key,[self.rnn_hidden_size])
        self.output_params=output_forward_trf.init(self.key,sample_h)
        self.output_forward=jit(output_forward_trf.apply)
        def loss_fn(params,hidden_state,target):
            pred=self.output_forward(params,hidden_state)
            return ((target-pred)**2).sum()
        self.loss_fn=jit(loss_fn)
        #Initialize the optimizers
        if optimizer=='sgd':
            print("Using optimizer SGD.")
            self.optimizer_rnn=optax.sgd(lr)
            self.optimizer_output=optax.sgd(lr)
        elif optimizer=='adam':
            print("Using optimizer ADAM.")
            self.optimizer_rnn=optax.adam(lr)
            self.optimizer_output=optax.adam(lr)
        self.optimizer_rnn_state=self.optimizer_rnn.init(self.rnn_params)
        self.optimizer_output_state=self.optimizer_output.init(self.output_params)
        self.rnn_sensitivity_fn=MultiplicativeRNN.sensitivity

    @abc.abstractmethod
    def step(self,obs,last_act,target):
        """

        Args:
            obs ([type]): [description]
            last_act ([type]): [description]
            target ([type]): [description]
        """

class CWPRNNPredictor(CompassWorldPredictor):
    def __init__(self, obs_size, act_size, truncation, rnn_hidden_size=40, hidden_size=32, lr=0.001, optimizer='sgd',seed=0) -> None:
        super().__init__(obs_size, act_size, truncation, rnn_activation_type='tanh',rnn_hidden_size=rnn_hidden_size, 
                        hidden_size=hidden_size,lr=lr, optimizer=optimizer,seed=seed)
        @jit #To make all operations jitable, also ensure that all non-static variables are passed as arguments
        def update(rnn_params,output_params,last_state,obs,last_act,target):
            #Pass through RNN and calculate sensitivities
            hidden_state,last_state=self.rnn_forward(rnn_params,obs,last_act,last_state)
            rnn_sensitivities=self.rnn_sensitivity_fn(self.rnn_forward,rnn_params,last_state)
            
            grad_fn=value_and_grad(self.loss_fn)
            #Calculate the gradients for the output layer and its sensitivites
            loss,grad_output_params=grad_fn(output_params,hidden_state,target)
            pred=self.output_forward(output_params,hidden_state)
            output_sensitivities=jax.jacrev(self.loss_fn,argnums=1)(output_params,hidden_state,target)
            #Calculate gradients by backpropagating sensitivities of output layer (chain rule)
            grad_rnn_params=jax.tree_multimap(lambda x: jnp.tensordot(output_sensitivities,x,axes=1), rnn_sensitivities) 
            return grad_rnn_params,grad_output_params,loss,pred,last_state
        self.update=update

    def step(self,obs,last_act,target):
        obs=jnp.array(CompassWorld.vectorize_color(obs))
        last_act=jnp.array(CompassWorld.vectorize_action(last_act))
        target=jnp.array(target)
        grad_rnn_params,grad_output_params,loss,pred,self.last_state=self.update(self.rnn_params,self.output_params,self.last_state,
                                                                                obs,last_act,target)
        updates, self.optimizer_rnn_state = self.optimizer_rnn.update(grad_rnn_params, self.optimizer_rnn_state)
        self.rnn_params=optax.apply_updates(self.rnn_params,updates)
        updates, self.optimizer_output_state = self.optimizer_output.update(grad_output_params, self.optimizer_output_state)
        self.output_params=optax.apply_updates(self.output_params,updates)
        return loss,pred


class GVFNTDPredictor(CompassWorldPredictor):
    def __init__(self, obs_size, act_size, truncation, rnn_hidden_size=40, hidden_size=32, lr=0.001,optimizer='sgd', seed=0) -> None:
        super().__init__(obs_size, act_size, truncation, rnn_activation_type='sigmoid',rnn_hidden_size=rnn_hidden_size, 
                        hidden_size=hidden_size, lr=lr, optimizer=optimizer,seed=seed)
        colors=['g','o','b','r','y']
        gammas=[1-2**k for k in range(-8,0)]
        self.gvfs=[initialize_gvf(TerminatingHorizonGVF,self.key,self.sample_o,self.sample_a,color,gamma,)  for color in colors for gamma in gammas]
        self.horde=Horde(self.gvfs)
        @jit
        def update(rnn_params,output_params,last_state,o_t,a_tminus1,o_tplus1,a_t, c_tplus1,gamma_tplus1,mu_otat, pi_otat, target_t):
            h_t,s_t=self.rnn_forward(rnn_params,o_t,a_tminus1,last_state)
            h_tplus1,s_tplus1=self.rnn_forward(rnn_params,o_tplus1,a_t,s_t)
            phi_t=self.rnn_sensitivity_fn(self.rnn_forward,rnn_params,s_t)
            delta_t=c_tplus1+gamma_tplus1*h_tplus1-h_t
            rho_t=jnp.exp(jnp.log(pi_otat)-jnp.log(mu_otat))
            grad_rnn_params=jax.tree_map(lambda x: -jnp.tensordot(rho_t*delta_t,x,axes=1), phi_t)
            grad_fn=value_and_grad(self.loss_fn)
            #Calculate the gradients for the output layer and its sensitivites
            loss,grad_output_params=grad_fn(output_params,h_t,target_t)
            pred=self.output_forward(output_params,h_t)
            return grad_rnn_params,grad_output_params,loss,pred,s_t
        self.update=update



    def step(self,o_t,a_tminus1,o_tplus1,a_t, mu_otat, target_t):
        """

        Args:
            o_t ([type]): Observation at timestep t
            a_tminus1 ([type]): Action at timestep t-1
            o_tplus1 ([type]): Observation at timestep t+1
            a_t ([type]): Action at timestep t
            target_t ([type]): Target at timestep t
        """
        o_t=jnp.array(CompassWorld.vectorize_color(o_t))
        a_tminus1=jnp.array(CompassWorld.vectorize_action(a_tminus1))
        o_tplus1=jnp.array(CompassWorld.vectorize_color(o_tplus1))
        a_t=jnp.array(CompassWorld.vectorize_action(a_t))
        mu_otat=jnp.array(mu_otat)
        target_t=jnp.array(target_t)
        #Calculate the GVF cumulants, gammas and policies
        pi_otat,c_tplus1,gamma_tplus1=self.horde.step(o_tplus1,a_t)
        #Call the update function
        grad_rnn_params,grad_output_params,loss,pred,self.last_state=self.update(self.rnn_params,self.output_params,self.last_state,o_t,a_tminus1,o_tplus1,a_t,c_tplus1,gamma_tplus1,
                    mu_otat,pi_otat,target_t)
        updates, self.optimizer_rnn_state = self.optimizer_rnn.update(grad_rnn_params, self.optimizer_rnn_state)
        self.rnn_params=optax.apply_updates(self.rnn_params,updates)
        updates, self.optimizer_output_state = self.optimizer_output.update(grad_output_params, self.optimizer_output_state)
        self.output_params=optax.apply_updates(self.output_params,updates)
        return loss,pred
        


        
        

        


