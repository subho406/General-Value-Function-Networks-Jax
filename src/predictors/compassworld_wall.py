import jax 
import haiku as hk
import optax
import jax.numpy as jnp

from src.models import MultiplicativeRNN
from jax import random,jit, value_and_grad

class CompassWorldPRNNWallPredictor:
    def __init__(self,obs_size,act_size,truncation,
                rnn_hidden_size=40,hidden_size=32,lr=0.001,
                seed=0) -> None:
        key=random.PRNGKey(seed)
        self.rnn_hidden_size=rnn_hidden_size
        self.hidden_size=hidden_size
        self.obs_size=obs_size
        self.act_size=act_size
        self.truncation=truncation
        #Initialize the RNN layer
        def rnn_forward(obs,last_act,last_state):
            rnn=MultiplicativeRNN(self.obs_size,self.act_size,self.rnn_hidden_size)
            out,state=rnn(obs,last_act,last_state)
            return out, state
        self.last_state=MultiplicativeRNN.initial_state(self.obs_size,self.act_size,self.rnn_hidden_size,
                                                        self.truncation)
        rnn_forward_trf=hk.without_apply_rng(hk.transform(rnn_forward))
        sample_o=jax.random.normal(key,[self.obs_size])
        sample_a=jax.random.normal(key,[self.act_size])
        self.rnn_params=rnn_forward_trf.init(key,sample_o,sample_a,self.last_state)
        self.rnn_forward=jit(rnn_forward_trf.apply)
        #Initilize the output layers
        def output_forward(hidden_state):
            output_layer=hk.Sequential([hk.Linear(self.hidden_size),jax.nn.relu,
                            hk.Linear(5)])
            pred=output_layer(hidden_state)
            return pred
        output_forward_trf=hk.without_apply_rng(hk.transform(output_forward))
        sample_h=jax.random.normal(key,[self.rnn_hidden_size])
        self.output_params=output_forward_trf.init(key,sample_h)
        self.output_forward=jit(output_forward_trf.apply)
        def loss_fn(params,hidden_state,target):
            pred=self.output_forward(params,hidden_state)
            return ((target-pred)**2).sum()
        self.loss_fn=jit(loss_fn)
        #Initialize the optimizers
        self.optimizer_rnn=optax.adam(lr)
        self.optimizer_rnn_state=self.optimizer_rnn.init(self.rnn_params)
        self.optimizer_output=optax.adam(lr)
        self.optimizer_output_state=self.optimizer_output.init(self.output_params)
        self.sensitivity_fn=MultiplicativeRNN.sensitivity

    def step(self,obs,last_act,target):
        obs=jnp.array(obs)
        last_act=jnp.array(last_act)
        target=jnp.array(target)
        #Pass through RNN and calculate sensitivities
        hidden_state,self.last_state=self.rnn_forward(self.rnn_params,obs,last_act,self.last_state)
        rnn_sensitivities=self.sensitivity_fn(self.rnn_forward,self.rnn_params,self.last_state)
        
        grad_fn=value_and_grad(self.loss_fn)
        #Calculate the gradients and apply update
        loss,grad_output_params=grad_fn(self.output_params,hidden_state,target)
        pred=self.output_forward(self.output_params,hidden_state)
        output_sensitivities=jax.jacrev(self.loss_fn,argnums=1)(self.output_params,hidden_state,target)
        #Calculate gradients
        grad_rnn_params=jax.tree_multimap(lambda x: jnp.tensordot(output_sensitivities,x,axes=1), rnn_sensitivities) 
        updates, self.optimizer_rnn_state = self.optimizer_rnn.update(grad_rnn_params, self.optimizer_rnn_state)
        self.rnn_params=optax.apply_updates(self.rnn_params,updates)
        updates, self.optimizer_output_state = self.optimizer_output.update(grad_output_params, self.optimizer_output_state)
        self.output_params=optax.apply_updates(self.output_params,updates)
        return loss,pred
        
        
        
        


