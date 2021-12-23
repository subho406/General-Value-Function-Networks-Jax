import haiku as hk
import jax.numpy as jnp

from typing import List
from jax import jit
from .gvf import GVFObject

class Horde:
    def __init__(self,gvfs:List[GVFObject]):
        self.gvfs=gvfs
        self.params=[gvf.params for gvf in gvfs]
        self.apply_fn=[gvf.apply_fn for gvf in gvfs]
        self.last_states=[gvf.class_def.initial_state() for gvf in gvfs]
        @jit
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
                
    
    def step(self,obs,last_act):
        """[summary]

        Args:
            obs ([type]): [description]
            last_act ([type]): [description]

        Returns:
            [jnp.array,jnp.array,jnp,array]: Returns concatenated pi(act|prev_state), C(obs|act,prev_state), gamma(obs|act,prev_state)
        """
        policies,cumulants,gammas,self.last_states=self.forward(self.params,self.last_states,obs,last_act)
        return policies,cumulants,gammas
        
