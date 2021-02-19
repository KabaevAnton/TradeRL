import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from enum import Enum

class Actions(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2





class SimpleTradingEnv(py_environment.PyEnvironment):

    def __init__(self, data: pd.DataFrame, history_size=30, time_steps_count=300):
        self.data = data
        self._current_time_step = 0
        self.deposit = 0
        self.in_shares = False
        

    def reset(self):
        """Return initial_time_step."""
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """Apply action and return new time_step."""
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(action)
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step

    def get_price(self):
        return 0
    

    def time_step_spec(self):
        """Return time_step_spec."""
        pass

    
    def observation_spec(self):
        """Return observation_spec."""
        pass

    
    def action_spec(self):
        """Return action_spec."""
        pass


    
    def _reset(self):
        """Return initial_time_step."""
        self._current_time_step = 0
        self.deposit = 0
        self.in_shares = False

    
    def _step(self, action: Actions):
        """Apply action and return new time_step."""
        if self._current_time_step > len(data):
            self.reset()

        if action == Actions.BUY:
            if not self.in_shares:
                self.deposit -= self.get_price()
                self.in_shares = True
        
        if action == Actions.SELL:
            if self.in_shares:
                self.deposit += self.get_price()
                self.in_shares = False

        if self._current_time_step == len(data) - 1:
            if self.in_shares:
                self.deposit += self.get_price()
            reward = np.log(self.deposit)
            return ts.termination(self.data[self._current_time_step], reward)
        else:
            return ts.transition(self.data[self._current_time_step], reward=0.0, discount=1.0)
            

        
        




    
