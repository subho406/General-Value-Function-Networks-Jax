import numpy as np


class CompassWorldRandomPolicy:
    actions=['l','r','f']

    def __init__(self,seed=0) -> None:
        self.rng=np.random.default_rng(seed=seed)
        self.reset()
    
    def reset(self):
        self.last_state=None
        self.leap=False


    def step(self,obs):
        if self.leap:
            if obs!='w': #We just hit a wall
                self.leap=False
                return self.rng.choice(self.actions)
            else:
                return 'f'
        else:
            if obs!='w' and self.rng.random()<0.1:
                self.leap=True
                return 'f'
            else:
                return self.rng.choice(self.actions)
