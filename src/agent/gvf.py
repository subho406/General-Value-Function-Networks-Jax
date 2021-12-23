import abc
import jax.numpy as jnp
import haiku as hk

from typing import Tuple, Any,Callable,NamedTuple,Optional

class GVF(hk.RNNCore):
    @abc.abstractmethod
    def __call__(self, obs, act, prev_state) -> Tuple[jnp.array,jnp.array,jnp.array, Any]:
        """The haiku GVF call function implements a functionality similar to the RNN interface, 
            but returns policy probability, cumulant, gamma

        Args:
            obs ([type]): [description]
            act ([type]): [description]
            prev_state ([type]): [description]

        Returns:
            Tuple[jnp.array, jnp.array, jnp.array, Any]: Returns pi(act|prev_state), C(obs|act,prev_state), gamma(obs|act,prev_state), new_state
        """

    @abc.abstractmethod
    def initial_state():
        """
            Might be needed in future, if GVF is composite or a neural network
        Args:
            batch_size (Optional[int]): [description]

        Returns:
            [type]: [description]
        """

class GVFObject(NamedTuple):
    params:Any
    apply_fn:Callable
    class_def:GVF


def initialize_gvf(gvf:GVF,key,sample_obs,sample_act,*args,**kwargs):
    """[summary]

    Args:
        gvf (GVF): [description]
        key ([type]): [description]
        sample_obs ([type]): [description]
        sample_act ([type]): [description]
    """
    def forward(obs,act,prev_state):
        network=gvf(*args,**kwargs)
        return network(obs,act,prev_state)
    gvf_trf=hk.without_apply_rng(hk.transform(forward))
    params=gvf_trf.init(key,sample_obs,sample_act,gvf.initial_state())
    return GVFObject(class_def=gvf,params=params,apply_fn=gvf_trf.apply)