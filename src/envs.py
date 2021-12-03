import numpy as np

class CompassWorld:
    color_codes={
            'w':0,
            'g':1,
            'o':2,
            'b':3,
            'r':4,
            'y':5
        }
    orientation_codes={
        'n':0,
        'e':1,
        's':2,
        'w':3
    }


    def __init__(self,height=8,width=8,seed=0) -> None:
        self.height=height
        self.width=width
        self.env=np.ones((height,width),dtype=np.int)*self.color_codes['w']
        self.env[0,:]=self.color_codes['o']
        self.env[:,self.width-1]=self.color_codes['y']
        self.env[self.height-1,:]=self.color_codes['r']
        self.env[:,0]=self.color_codes['b']
        self.env[1,0]=self.color_codes['g']
        self.rng=np.random.default_rng(seed=seed)
        self.reset()

    def reset(self):
        self.agent_pos=np.concatenate([self.rng.integers(1,self.height-1,size=1),self.rng.integers(1,self.width-1,size=1)])
        self.agent_orientation=self.rng.integers(0,4,size=1)
    
    def step(self,action):
        if action=='l': #Take left action
            self.agent_orientation=(self.agent_orientation-1)%4
        elif action=='r': #Take the right action
            self.agent_orientation=(self.agent_orientation+1)%4
        elif action=='f': #Move forward
            if self.agent_orientation==0:
                self.agent_pos[0]=max(1,self.agent_pos[0]-1)
            elif self.agent_orientation==1:
                self.agent_pos[1]=min(self.width-2,self.agent_pos[1]+1)
            elif self.agent_orientation==2:
                self.agent_pos[0]=min(self.height-2,self.agent_pos[0]+1)
            elif self.agent_orientation==3:
                self.agent_pos[1]=max(1,self.agent_pos[1]-1)
    
    def full_state(self):
        """
            Return the fully observable state (current_state,last_action) pair
        """
        pass    

    def observe(self):
        """
            Return the partially observable vector
        """
        pass



env=CompassWorld(8,8)
print(env.env)
print(env.agent_pos)