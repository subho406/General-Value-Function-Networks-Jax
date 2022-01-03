import haiku as hk
import jax.numpy as jnp

from typing import List
from jax import jit
from .gvf import GVFObject
from functools import partial

class Horde:
    def __init__(self,gvfs:List[GVFObject]):
        self.gvfs=gvfs
        self.apply_fn=[gvf.apply_fn for gvf in gvfs]
        def forward(params,last_states,obs,last_act):
            cumulants=jnp.zeros(len(self.gvfs))
            policies=jnp.zeros(len(self.gvfs))
            gammas=jnp.zeros(len(self.gvfs))
            for i in range(len(self.gvfs)):
                policy,cumulant,gamma,last_states[i]=self.apply_fn[i](params[i],obs,last_act,last_states[i])
                policies=policies.at[i].set(policy)
                cumulants=cumulants.at[i].set(cumulant)
                gammas=gammas.at[i].set(gamma)
            return policies,cumulants,gammas,last_states
        self.forward=forward
    
    def initialize(self):
        params=[gvf.params for gvf in self.gvfs]
        last_state=[gvf.class_def.initial_state() for gvf in self.gvfs]
        return params,last_state
    
    @partial(jit, static_argnums=(0,)) 
    def forward(self,params,last_state,obs,last_act):
        """[summary]

        Args:
            obs ([type]): [description]
            last_act ([type]): [description]

        Returns:
            [jnp.array,jnp.array,jnp,array]: Returns concatenated pi(act|prev_state), C(obs|act,prev_state), gamma(obs|act,prev_state)
        """
        policies,cumulants,gammas,new_state=self.forward(params,last_state,obs,last_act)
        return policies,cumulants,gammas,new_state
        
