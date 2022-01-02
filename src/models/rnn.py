from haiku._src.module import Kwargs
import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp

from functools import partial
from jax import jit,custom_jvp
from src.utils.utils import tree_dot
from haiku import Transformed
from haiku._src.recurrent import add_batch
from typing import Optional, Tuple, Any, NamedTuple
from jax import lax

class BasePRNN(hk.RNNCore): 

    @staticmethod
    @partial(jit, static_argnums=(0,)) 
    def hvp(rnn_forward,rnn_params,inputs,last_state,vector):
        def jvp(rnn_params,inputs,last_state,vector):
            jacobians=jax.jacrev(rnn_forward)(rnn_params,inputs,last_state)[0]
            jvp=tree_dot(jacobians,vector)
            return jvp
        hvp_fn=jax.jacrev(jvp)
        return hvp_fn(rnn_params,inputs,last_state,vector)

    @staticmethod
    @partial(jit, static_argnums=(0,)) 
    def sensitivity(rnn_forward,rnn_params,rnn_state):
        """Returns the jacobian of the current hidden state with respect to the RNN params 
            until the length of the trajectory. 

            In future this will be replaced with with JAX grad primitives
        Args:
            rnn_forward_pure ([type]): Pure haiku RNN transformed function 
            rnn_params ([type]): RNN parameters for the transformed function
            rnn_trajectory (tuple(jnp.array,jnp.array)): Tuple of (inputs,last hidden_state)

        Returns:
            jax.numpy.array: Tensor containing the sensitivities as a Jacobian matrix
        """
        hidden_state,trajectory=rnn_state
        last_hidden_states=trajectory.last_hidden_states
        observations=trajectory.observations
        last_actions=trajectory.last_actions
        
        def rnn_forward_out(params,obs, act,last_hidden_state):
            out,_=rnn_forward(params,(obs,act),(last_hidden_state,None))
            return out #We do not need the state output hence we do this trick 


        rnn_jac_theta=jax.jacrev(rnn_forward_out)
        rnn_jac_hidden=jax.jacrev(rnn_forward_out,argnums=3)
        del_h_tminus1_theta=rnn_jac_theta(rnn_params,observations[0],last_actions[0],last_hidden_states[0])
        del_h_t_theta=del_h_tminus1_theta

        def sensitivity_calc(del_h_tminus1_theta,trajectory):
            o_t,a_tminus1,h_tminus1=trajectory
            del_f_theta=rnn_jac_theta(rnn_params,o_t,a_tminus1,h_tminus1)
            del_f_h_tminus1=rnn_jac_hidden(rnn_params,o_t,a_tminus1,h_tminus1)
            # del_h_t_theta=del_f_h_tminus1*del_h_tminus1_theta
            del_h_t_theta= jax.tree_map(lambda x: jnp.tensordot(del_f_h_tminus1,x,axes=1), del_h_tminus1_theta) 
            # del_h_t_theta+=del_f_theta
            del_h_t_theta=jax.tree_multimap(lambda x, y: x+y, del_h_t_theta, del_f_theta)
            return del_h_t_theta,None

        def scan_all_prev():
            scan_all_prev,_=jax.lax.scan(sensitivity_calc,del_h_tminus1_theta,(observations[1:],last_actions[1:],
                                                                                last_hidden_states[1:]))
            return scan_all_prev
            
        del_h_t_theta=lax.cond(observations.shape[0]>1,lambda x: scan_all_prev(),lambda x: del_h_t_theta,None) #Execute the expesive forward prop rule only if truncation is greater than 1                                                                       
        return del_h_t_theta

        


class TrajectoryState(NamedTuple):
    """Trajectory State to store last p transactions

    Args:
        NamedTuple ([type]): [description]
    """
    last_hidden_states:jnp.ndarray
    observations:jnp.ndarray
    last_actions:jnp.ndarray


class MultiplicativeRNN(BasePRNN):
    def __init__(self, obs_size,action_size,hidden_size,activation='sigmoid',name: Optional[str] = None):
        """Implements a Multiplicative RNN on the observation and action inputs as described in: (paper in review) 

        Args:
            obs_size ([type]): [description]
            action_size ([type]): [description]
            hidden_size ([type]): [description]
            name (Optional[str], optional): [description]. Defaults to None.
        """
        super().__init__(name=name)
        self.input_size=obs_size
        self.action_size=action_size
        self.hidden_size=hidden_size
        self.activation=activation
    
    def __call__(self, inputs,  prev_state) -> Tuple[Any, Any]:
        obs,act=inputs
        prev_hidden_state,trajectory=prev_state
        #Take one step in RNN 
        #glorot_uniform initializer
        init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')

        w_o=hk.get_parameter("w_o",[self.hidden_size,self.input_size,self.action_size],init=init)
        w_h=hk.get_parameter("w_h",[self.hidden_size,self.hidden_size,self.action_size],init=init)
        b=hk.get_parameter("b",[self.hidden_size,self.action_size],init=init) #The bias conditioned on action
        out_h=jnp.tensordot(jnp.tensordot(w_h,prev_hidden_state,axes=(1,0)),act,axes=1)
        out_o=jnp.tensordot(jnp.tensordot(w_o,obs,axes=(1,0)),act,axes=1)
        bias=jnp.tensordot(b,act,axes=1)
        if self.activation=='sigmoid':
            out=jax.nn.sigmoid(out_h+out_o+bias)
        elif self.activation=='tanh':
            out=jax.nn.tanh(out_h+out_o+bias)
        #Update the trajectories array
        if trajectory is not None:
            last_hidden_states=jnp.concatenate((trajectory.last_hidden_states[1:],
                                            prev_hidden_state.reshape(1,-1)),axis=0)
            observations=jnp.concatenate((trajectory.observations[1:],obs.reshape(1,-1)),axis=0)
            last_actions=jnp.concatenate((trajectory.last_actions[1:],act.reshape(1,-1)),axis=0)
            

            state=(out,TrajectoryState(observations=observations,
                        last_hidden_states=last_hidden_states,
                        last_actions=last_actions))
        else:
            state=(out,None)
        return out,state 

    @staticmethod
    def initial_state(obs_size,action_size,hidden_size,truncation,batch_size: Optional[int] = None):
        state = (jnp.zeros([hidden_size]),
                    TrajectoryState(observations=jnp.zeros([truncation,obs_size]),
                                    last_hidden_states=jnp.zeros([truncation,hidden_size]),
                                    last_actions=jnp.zeros([truncation,action_size])))
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state
    
    
def rnn_transform(rnn_class:BasePRNN,*args,**kwargs):
    """Returns a haiku transformed pure function for the RNN with p-sensitivity gradients
    """
    def rnn_forward(inputs,last_state):
        rnn=rnn_class(*args,**kwargs)
        out,state=rnn(inputs,last_state)
        return out, state
    transformed=hk.without_apply_rng(hk.transform(rnn_forward))
    forward_fn=hk.without_apply_rng(hk.transform(rnn_forward)).apply #We need a copy to use it in autodiff calculutations inside custom jvp
    transformed_f=custom_jvp(transformed.apply)
    def f_jvp(primals, tangents):
        rnn_params,inputs,last_state=primals
        rnn_params_t,inputs_t,last_state_t=tangents
        primal_out=transformed_f(rnn_params,inputs,last_state)
        jacobian=rnn_class.sensitivity(forward_fn,rnn_params,primal_out[1])
        tangent_out=jax.tree_multimap(lambda x, y: jnp.tensordot(x,y,y.ndim), jacobian, rnn_params_t)
        tangent_out=jnp.stack(jax.tree_util.tree_flatten(tangent_out)[0],axis=0).sum(axis=0)
        return primal_out,(tangent_out,primal_out[1]) #Second parameter currently returns the primal_out (forward propagation)
    transformed_f.defjvp(f_jvp)
    return Transformed(init=transformed.init,apply=transformed_f)




   