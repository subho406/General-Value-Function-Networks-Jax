import jax
import optax
import jax.numpy as jnp

from jax import jit,random
from src.agent.gvf import GVF
from src.models.rnn import MultiplicativeRNN, rnn_transform
from functools import partial
from typing import List
from .gvfn import Horde

class GVFN_TD:
    def __init__(self,obs_size,act_size,truncation,gvfs:List[GVF],rnn_activation_type='sigmoid',
                        lr=0.001,optimizer='sgd',seed=0) -> None:
        self.key=random.PRNGKey(seed)
        self.obs_size=obs_size
        self.act_size=act_size
        self.truncation=truncation
        self.rnn_activation_type=rnn_activation_type
        self.hidden_size=len(gvfs)
        self.optimizer=optimizer
        self.sample_o=jax.random.normal(self.key,[self.obs_size])
        self.sample_a=jax.random.normal(self.key,[self.act_size])
        if optimizer=='sgd':
            self.optimizer=optax.sgd(lr)
        elif optimizer=='adam':
            self.optimizer=optax.adam(lr)
        self.rnn_sensitivity_fn=jit(jax.jacrev(self.rnn_forward))
        self.horde=Horde(self.gvfs)
        
    def initialize(self):
        """Initializes a GVFN Network with parameters, optimizer state and last_state

        Returns:
            [type]: [description]
        """
        self.rnn_forward_trf=rnn_transform(MultiplicativeRNN,self.obs_size,self.act_size,self.hidden_size,
                                activation=self.rnn_activation_type)
        last_rnn_state=MultiplicativeRNN.initial_state(self.obs_size,self.act_size,self.hidden_size,
                                                         self.truncation)
        params=self.rnn_forward_trf.init(self.key,(self.sample_o,self.sample_a),self.last_state)
        self.rnn_forward=jit(self.rnn_forward_trf.apply)
        optimizer_state=self.optimizer.init(params)
        horde_params,horde_last_state=self.horde.initialize()
        last_state=(last_rnn_state,horde_params,horde_last_state)
        return params,optimizer_state,last_state
    
    @partial(jit, static_argnums=(0,)) 
    def forward(self,params, optimizer_state, inputs,last_state):
        out={}
        last_rnn_state,horde_params,horde_last_state=last_state
        inputs['pi_otat_output'],inputs['c_t+1_output'],inputs['gamma_t+1_output'],_=self.horde.forward(horde_params,horde_last_state,
                                                                                    inputs['o_t+1'],inputs['a_t'])
        out['h_t'],s_t=self.rnn_forward(params,(inputs['o_t'],inputs['a_t-1']),last_rnn_state)
        out['h_t+1'],s_tplus1=self.rnn_forward(params,(inputs['o_t+1'],inputs['a_t']),s_t)
        #Calculate GVFN TD gradients
        𝝫_t,_=self.rnn_sensitivity_fn(params,(inputs['o_t'],inputs['a_t-1']),last_rnn_state)
        δ_t=inputs['c_t+1']+inputs['gamma_t+1']*out['h_t+1']-out['h_t']
        ρ_t=jnp.exp(jnp.log(inputs['pi_otat'])-jnp.log(inputs['mu_otat']))
        grad_rnn_params=jax.tree_map(lambda x: -jnp.tensordot(ρ_t*δ_t,x,axes=1), 𝝫_t)
        updates, optimizer_state = self.optimizer.update(grad_rnn_params, optimizer_state)
        params=optax.apply_updates(params,updates)
        new_state=(s_t,horde_params,horde_last_state)
        return params,optimizer_state,new_state,out
        
