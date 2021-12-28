"""
    Implements the Compass World env
"""
from os import stat
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

    action_codes={
        'l':0,
        'r':1,
        'f':2
    }

    def __init__(self,height=8,width=8,seed=0) -> None:
        self.height=height+2 #Add the length of the boundary
        self.width=width+2 
        self.env=np.chararray((self.height,self.width))
        self.env[:]='w'
        self.env[0,:]='o'
        self.env[:,self.width-1]='y'
        self.env[self.height-1,:]='r'
        self.env[:,0]='b'
        self.env[1,0]='g'
        self.rng=np.random.default_rng(seed=seed)
        self.reset()

    def reset(self):
        self.agent_pos=np.concatenate([self.rng.integers(1,self.height-1,size=1),self.rng.integers(1,self.width-1,size=1)])
        self.agent_orientation=self.rng.integers(0,4,size=1)
    
    def step(self,action):
        """Takes an step in the compass world environment

        Args:
            action (str): The action to take, one of ('l','r','f')

        Returns:
            str: The color observed by the agent
        """
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
        #Get the observation 
        return self.observe()
    
    def full_state(self):
        """
            Return the fully observable state (current_state,last_action) pair
        """
        return self.agent_pos,self.agent_orientation  


    def observe(self):
        """
            Return the observed color as 
        """
        if self.agent_orientation==0:
            return str(self.env[self.agent_pos[0]-1,self.agent_pos[1]],'utf-8')
        elif self.agent_orientation==1:
            return str(self.env[self.agent_pos[0],self.agent_pos[1]+1],'utf-8')
        elif self.agent_orientation==2:
            return str(self.env[self.agent_pos[0]+1,self.agent_pos[1]],'utf-8')
        elif self.agent_orientation==3:
            return str(self.env[self.agent_pos[0],self.agent_pos[1]-1],'utf-8')
    
    def wall_ahead(self):
        """Return the color of the wall in front of the agent
        """
        if self.agent_orientation==0:
            color=str(self.env[0,self.agent_pos[1]],'utf-8')
        elif self.agent_orientation==1:
            color=str(self.env[self.agent_pos[0],self.width-1],'utf-8')
        elif self.agent_orientation==2:
            color=str(self.env[self.height-1,self.agent_pos[1]],'utf-8')
        elif self.agent_orientation==3:
            color=str(self.env[self.agent_pos[0],0],'utf-8')
        return color


    @staticmethod
    def vectorize_color(color):
        """Returns a numpy array where a color is encoded with two bits per color: one to indicate the agent observes that color,
                and the other to indicate another color is observed

        Args:
            color (str): Color 

        Returns:
            np.array : [description]
        """
        color_idx=CompassWorld.color_codes[color]
        color_vec=np.array([1,0]*len(CompassWorld.color_codes))
        color_vec[color_idx*2]=0
        color_vec[color_idx*2+1]=1
        return color_vec

    @staticmethod
    def vectorize_action(action):
        action_idx=CompassWorld.action_codes[action]
        action_vec=np.zeros(len(CompassWorld.action_codes))
        action_vec[action_idx]=1
        return action_vec
    
    @staticmethod
    def action_vec_size():
        return len(CompassWorld.action_codes)
    
    @staticmethod
    def color_vec_size():
        return len(CompassWorld.color_codes)*2
    

