import jax 
import abc
import haiku as hk
import optax
import jax.numpy as jnp
import numpy as np
from src.agent.gvfn import GVFN

from src.models.rnn import MultiplicativeRNN, rnn_transform
from src.agent import Horde,initialize_gvf
from src.utils.utils import tree_dot,tree_sum
from src.envs import CompassWorld
from jax import random,jit, value_and_grad
from .gvfs import TerminatingHorizonGVF



class CompassWorldPredictor:
    colors=list(CompassWorld.color_codes.keys())[1:]
    colors_to_idx={}
    for i, c in enumerate(colors): colors_to_idx[c]=i

    @staticmethod
    def vectorize_target(color):
        color_idx=CompassWorld.color_codes[color]
        wall_vec=np.zeros(len(CompassWorld.color_codes)-1)
        wall_vec[color_idx-1]=1
        return wall_vec

    def __init__(self,obs_size,act_size,truncation,rnn_activation_type='sigmoid',
                rnn_hidden_size=40,hidden_size=32,lr=0.001,optimizer='sgd',
                seed=0) -> None:
        self.key=random.PRNGKey(seed)
        self.rnn_hidden_size=rnn_hidden_size
        self.hidden_size=hidden_size
        self.obs_size=obs_size
        self.act_size=act_size
        self.truncation=truncation
        self.rnn_activation_type=rnn_activation_type
        def output_forward(hidden_state):
            output_layer=hk.Sequential([hk.Linear(self.hidden_size),jax.nn.relu,
                            hk.Linear(5)])
            pred=output_layer(hidden_state)
            return pred
        output_forward_trf=hk.without_apply_rng(hk.transform(output_forward))
        sample_h=jax.random.normal(self.key,[self.rnn_hidden_size])
        self.output_params=output_forward_trf.init(self.key,sample_h)
        self.output_forward=jit(output_forward_trf.apply)
        def loss_fn(params,hidden_state,target,ρ_t_target):
            pred=self.output_forward(params,hidden_state)
            return (ρ_t_target*((pred-target)**2)).sum()*(1/(2*ρ_t_target.shape[0]))
        self.loss_fn=jit(loss_fn)
        #Initialize the optimizers
        if optimizer=='sgd':
            self.optimizer_output=optax.sgd(lr)
        elif optimizer=='adam':
            self.optimizer_output=optax.adam(lr)
        self.optimizer_output_state=self.optimizer_output.init(self.output_params)
        #Initialize the output GVFS
        self.output_gvfs=[initialize_gvf(TerminatingHorizonGVF,self.key,self.sample_o,self.sample_a,color,1.0,)  for color in self.colors]
        self.output_horde=Horde(self.output_gvfs)

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
        def update(rnn_params,output_params,last_state,optimizer_rnn_state,optimizer_output_state,inputs):
            #Pass through RNN and calculate sensitivities
            h_t,s_t=self.rnn_forward(rnn_params,(inputs['o_t'],inputs['a_t-1']),last_state)
            h_tplus1,s_tplus1=self.rnn_forward(rnn_params,(inputs['o_t+1'],inputs['a_t']),s_t)
            rnn_sensitivities,_=self.rnn_sensitivity_fn(rnn_params,(inputs['o_t'],inputs['a_t-1']),last_state)
            pred_t=self.output_forward(output_params,h_t)
            pred_tplus1=self.output_forward(output_params,h_tplus1)
            target=inputs['c_t+1_output']+inputs['gamma_t+1_output']*pred_tplus1
            ρ_t_target=jnp.exp(jnp.log(inputs['pi_otat_output'])-jnp.log(inputs['mu_otat']))
            grad_fn=value_and_grad(self.loss_fn)
            #Calculate the gradients for the output layer and its sensitivites
            loss,grad_output_params=grad_fn(output_params,h_t,target,ρ_t_target)
            output_sensitivities=jax.jacrev(self.loss_fn,argnums=1)(output_params,h_t,target,ρ_t_target) #Legacy: we replace with end-to-end loss_fn later.
            #Calculate gradients by backpropagating sensitivities of output layer (chain rule)
            grad_rnn_params=jax.tree_multimap(lambda x: jnp.tensordot(output_sensitivities,x,axes=1), rnn_sensitivities) 
            #Apply the calculated updates
            updates, optimizer_rnn_state = self.optimizer_rnn.update(grad_rnn_params, optimizer_rnn_state)
            rnn_params=optax.apply_updates(rnn_params,updates)
            updates, optimizer_output_state = self.optimizer_output.update(grad_output_params, optimizer_output_state)
            output_params=optax.apply_updates(output_params,updates)
            return rnn_params,output_params,optimizer_rnn_state,optimizer_output_state,loss,pred_t,s_t
        self.update=update

    def step(self,o_t,a_tminus1,o_tplus1,a_t, mu_otat):
        inputs={}
        inputs['o_t']=jnp.array(CompassWorld.vectorize_color(o_t))
        inputs['a_t-1']=jnp.array(CompassWorld.vectorize_action(a_tminus1))
        inputs['o_t+1']=jnp.array(CompassWorld.vectorize_color(o_tplus1))
        inputs['a_t']=jnp.array(CompassWorld.vectorize_action(a_t))
        inputs['mu_otat']=jnp.array(mu_otat)
        #Calculate output GVF 
        inputs['pi_otat_output'],inputs['c_t+1_output'],inputs['gamma_t+1_output']=self.output_horde.step(inputs['o_t+1'],inputs['a_t'])
        self.rnn_params,self.output_params,self.optimizer_rnn_state,self.optimizer_output_state,loss,pred,self.last_state=self.update(self.rnn_params,self.output_params,self.last_state,
                                                        self.optimizer_rnn_state,self.optimizer_output_state,inputs)
        return loss,pred


class GVFNTDPredictor(CompassWorldPredictor):
    def __init__(self, obs_size, act_size, truncation, rnn_hidden_size=40, hidden_size=32, lr=0.001,optimizer='sgd', seed=0) -> None:
        super().__init__(obs_size, act_size, truncation, rnn_activation_type='sigmoid',rnn_hidden_size=rnn_hidden_size, 
                        hidden_size=hidden_size, lr=lr, optimizer=optimizer,seed=seed)
        
        gammas=[1-2.0**k for k in range(-7,1)]
        self.gvfs=[initialize_gvf(TerminatingHorizonGVF,self.key,self.sample_o,self.sample_a,color,gamma,)  for color in self.colors for gamma in gammas]
        self.gvfn_network=GVFN()

        @jit
        def update(rnn_params,output_params,last_state,optimizer_rnn_state,optimizer_output_state,inputs):
            h_t,s_t=self.rnn_forward(rnn_params,(inputs['o_t'],inputs['a_t-1']),last_state)
            h_tplus1,s_tplus1=self.rnn_forward(rnn_params,(inputs['o_t+1'],inputs['a_t']),s_t)
            pred_t=self.output_forward(output_params,h_t)
            pred_tplus1=self.output_forward(output_params,h_tplus1)
            #Calculate GVFN TD gradients
            𝝫_t,_=self.rnn_sensitivity_fn(rnn_params,(inputs['o_t'],inputs['a_t-1']),last_state)
            δ_t=inputs['c_t+1']+inputs['gamma_t+1']*h_tplus1-h_t
            ρ_t=jnp.exp(jnp.log(inputs['pi_otat'])-jnp.log(inputs['mu_otat']))
            grad_rnn_params=jax.tree_map(lambda x: -jnp.tensordot(ρ_t*δ_t,x,axes=1), 𝝫_t)
            #Calculate output layer gradients
            target_t=inputs['c_t+1_output']+inputs['gamma_t+1_output']*pred_tplus1
            ρ_t_target=jnp.exp(jnp.log(inputs['pi_otat_output'])-jnp.log(inputs['mu_otat']))
            grad_fn=value_and_grad(self.loss_fn)
            loss,grad_output_params=grad_fn(output_params,h_t,target_t,ρ_t_target)
            #Apply the calculated updates
            updates, optimizer_rnn_state = self.optimizer_rnn.update(grad_rnn_params, optimizer_rnn_state)
            rnn_params=optax.apply_updates(rnn_params,updates)
            updates, optimizer_output_state = self.optimizer_output.update(grad_output_params, optimizer_output_state)
            output_params=optax.apply_updates(output_params,updates)
            return rnn_params,output_params,optimizer_rnn_state,optimizer_output_state,loss,pred_t,s_t
            
        self.update=update

    def step(self,o_t,a_tminus1,o_tplus1,a_t, mu_otat):
        """

        Args:
            o_t ([type]): Observation at timestep t
            a_tminus1 ([type]): Action at timestep t-1
            o_tplus1 ([type]): Observation at timestep t+1
            a_t ([type]): Action at timestep t
            target_t ([type]): Target at timestep t
        """
        inputs={}
        inputs['o_t']=jnp.array(CompassWorld.vectorize_color(o_t))
        inputs['a_t-1']=jnp.array(CompassWorld.vectorize_action(a_tminus1))
        inputs['o_t+1']=jnp.array(CompassWorld.vectorize_color(o_tplus1))
        inputs['a_t']=jnp.array(CompassWorld.vectorize_action(a_t))
        inputs['mu_otat']=jnp.array(mu_otat)
        #Calculate the GVF cumulants, gammas and policies
        inputs['pi_otat'],inputs['c_t+1'],inputs['gamma_t+1']=self.horde.step(inputs['o_t+1'],inputs['a_t'])

        inputs['pi_otat_output'],inputs['c_t+1_output'],inputs['gamma_t+1_output']=self.output_horde.step(inputs['o_t+1'],inputs['a_t'])

        self.rnn_params,self.output_params,self.optimizer_rnn_state,self.optimizer_output_state,loss,pred,self.last_state=self.update(self.rnn_params,self.output_params,
                                                        self.last_state,self.optimizer_rnn_state,self.optimizer_output_state,inputs)
        return loss,pred

class GVFNTDCPredictor(CompassWorldPredictor):
    #TDC Agent
    def __init__(self, obs_size, act_size, truncation, rnn_hidden_size=40, hidden_size=32, lr=0.001,optimizer='sgd', beta=0.001,seed=0) -> None:
        super().__init__(obs_size, act_size, truncation, rnn_activation_type='sigmoid',rnn_hidden_size=rnn_hidden_size, 
                        hidden_size=hidden_size, lr=lr, optimizer=optimizer,seed=seed)
        
        gammas=[1-2.0**k for k in range(-7,1)]
        self.gvfs=[initialize_gvf(TerminatingHorizonGVF,self.key,self.sample_o,self.sample_a,color,gamma,)  for color in self.colors for gamma in gammas]
        self.horde=Horde(self.gvfs)

        #We use the custom gradient version of transform here
        self.rnn_forward_trf=rnn_transform(MultiplicativeRNN,self.obs_size,self.act_size,self.rnn_hidden_size,
                                activation=self.rnn_activation_type)
        self.sample_o=jax.random.normal(self.key,[self.obs_size])
        self.sample_a=jax.random.normal(self.key,[self.act_size])
        
        self.rnn_params=self.rnn_forward_trf.init(self.key,(self.sample_o,self.sample_a),self.last_state)
        self.rnn_forward=jit(self.rnn_forward_trf.apply)
        #TDC weights
        self.key,subkey=random.split(self.key)
        self.w_params=self.rnn_forward_trf.init(subkey,(self.sample_o,self.sample_a),self.last_state)
        self.rnn_sensitivity_fn=jit(jax.jacrev(self.rnn_forward))
        self.rnn_hvp_fun=MultiplicativeRNN.hvp
        #TDC weights optimizer
        if optimizer=='sgd':
            self.optimizer_w=optax.sgd(lr)
        elif optimizer=='adam':
            self.optimizer_w=optax.adam(lr)
        self.optimizer_w_state=self.optimizer_w.init(self.w_params)

        @jit
        def update(rnn_params,w_params,output_params,last_state,optimizer_rnn_state,optimizer_w_state,optimizer_output_state,inputs):
            h_t,s_t=self.rnn_forward(rnn_params,(inputs['o_t'],inputs['a_t-1']),last_state)
            h_tplus1,s_tplus1=self.rnn_forward(rnn_params,(inputs['o_t+1'],inputs['a_t']),s_t)
            pred_t=self.output_forward(output_params,h_t)
            pred_tplus1=self.output_forward(output_params,h_tplus1)
            #Calculate GVFN TDC gradients
            𝝫_t=self.rnn_sensitivity_fn(rnn_params,(inputs['o_t'],inputs['a_t-1']),last_state)[0] #Jacobian of s_t
            _𝝫_t=self.rnn_sensitivity_fn(rnn_params,(inputs['o_t+1'],inputs['a_t']),s_t)[0] #Jacobian of s_t+1
            δ_t=inputs['c_t+1']+inputs['gamma_t+1']*h_tplus1-h_t
            ρ_t=jnp.exp(jnp.log(inputs['pi_otat'])-jnp.log(inputs['mu_otat']))
            v_t=MultiplicativeRNN.hvp(self.rnn_forward,rnn_params,(inputs['o_t'],inputs['a_t-1']),last_state,w_params)
            _δ_t=tree_dot(𝝫_t,w_params)
            Ψ_t=jax.tree_map(lambda x:jnp.tensordot(ρ_t*δ_t-_δ_t,x,axes=1),v_t)
            grad_rnn_term1=jax.tree_map(lambda x: -jnp.tensordot(ρ_t*δ_t,x,axes=1), 𝝫_t)
            grad_rnn_term2=jax.tree_map(lambda x: jnp.tensordot(ρ_t*inputs['gamma_t+1']*_δ_t,x,axes=1), _𝝫_t)
            grad_rnn_params=tree_sum(tree_sum(grad_rnn_term1,grad_rnn_term2),Ψ_t)
            #Calculate the w_params gradients
            
            grad_w_params=jax.tree_map(lambda x: -jnp.tensordot(ρ_t*(δ_t-_δ_t),x,axes=1), 𝝫_t)
            #Calculate output layer gradients
            target_t=inputs['c_t+1_output']+inputs['gamma_t+1_output']*pred_tplus1
            ρ_t_target=jnp.exp(jnp.log(inputs['pi_otat_output'])-jnp.log(inputs['mu_otat']))
            grad_fn=value_and_grad(self.loss_fn)
            loss,grad_output_params=grad_fn(output_params,h_t,target_t,ρ_t_target)
            #Apply the calculated updates
            updates, optimizer_rnn_state = self.optimizer_rnn.update(grad_rnn_params, optimizer_rnn_state)
            rnn_params=optax.apply_updates(rnn_params,updates) #RNN
            updates, optimizer_w_state= self.optimizer_w.update(grad_w_params, optimizer_w_state)
            w_params=optax.apply_updates(w_params,updates) #w params
            updates, optimizer_output_state = self.optimizer_output.update(grad_output_params, optimizer_output_state)
            output_params=optax.apply_updates(output_params,updates) #Output layer
            return rnn_params,w_params,output_params,optimizer_rnn_state,optimizer_w_state,optimizer_output_state,loss,pred_t,s_t
            
        self.update=update


    def step(self,o_t,a_tminus1,o_tplus1,a_t, mu_otat):
        """

        Args:
            o_t ([type]): Observation at timestep t
            a_tminus1 ([type]): Action at timestep t-1
            o_tplus1 ([type]): Observation at timestep t+1
            a_t ([type]): Action at timestep t
            target_t ([type]): Target at timestep t
        """
        inputs={}
        inputs['o_t']=jnp.array(CompassWorld.vectorize_color(o_t))
        inputs['a_t-1']=jnp.array(CompassWorld.vectorize_action(a_tminus1))
        inputs['o_t+1']=jnp.array(CompassWorld.vectorize_color(o_tplus1))
        inputs['a_t']=jnp.array(CompassWorld.vectorize_action(a_t))
        inputs['mu_otat']=jnp.array(mu_otat)
        #Calculate the GVF cumulants, gammas and policies
        inputs['pi_otat'],inputs['c_t+1'],inputs['gamma_t+1']=self.horde.step(inputs['o_t+1'],inputs['a_t'])

        inputs['pi_otat_output'],inputs['c_t+1_output'],inputs['gamma_t+1_output']=self.output_horde.step(inputs['o_t+1'],inputs['a_t'])

        self.rnn_params,self.w_params,self.output_params,self.optimizer_rnn_state,self.optimizer_w_state,self.optimizer_output_state,loss,pred,self.last_state=self.update(self.rnn_params,self.w_params,self.output_params,
                                                        self.last_state,
                                                        self.optimizer_rnn_state,self.optimizer_w_state,self.optimizer_output_state,inputs)
        return loss,pred
        


        
        

        


