import jax.numpy as jnp

from src.envs import CompassWorld
from src.agent import GVF
from jax import lax

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