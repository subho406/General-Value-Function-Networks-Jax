import numpy as np


class CompassWorldRandomPolicy:
    action_probs={'l':0.2,'r':0.2,'f':0.6}
    actions=list(action_probs.keys())
    probs=list(action_probs.values())

    def __init__(self,seed=0) -> None:
        self.rng=np.random.default_rng(seed=seed)
        self.reset()
    
    def reset(self):
        self.last_state=None
        self.leap=False
    
    def random_choice_action(self):
        action=self.rng.choice(self.actions,p=self.probs)
        return action,self.action_probs[action]

    def step(self,obs):
        if self.leap==False:
            if self.rng.random()>0.9:
                self.leap=True
        if self.leap:
            if obs!='w': #We just hit a wall
                self.leap=False
            else:
                return 'f', 1.0
        return self.random_choice_action()
