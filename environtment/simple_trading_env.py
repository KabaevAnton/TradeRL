if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath('./'))

import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from enum import Enum

from common.market import buy_shares, sell_shares

class Actions(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2




class SimpleTradingEnv(py_environment.PyEnvironment):

    def __init__(self, data: np.array, deposit=10000.0, history_len=0, close_price_idx=3):
        super().__init__()
        self.data = data
        self.data_index = 0
        self.start_deposit = deposit
        self.deposit = deposit
        self.shares = 0
        self.history_len = history_len
        self.close_price_idx = close_price_idx  
              
        
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(history_len + 1,data.shape[1]), dtype=np.float32, name='observation')       
        

         

       

    def get_price(self):
        return self.data[self.data_index][self.close_price_idx]      

    
    def observation_spec(self):
        """Return observation_spec."""
        return self._observation_spec
    
    def action_spec(self):
        """Return action_spec."""
        return self._action_spec

    def get_state(self):
        start = self.data_index - self.history_len
        start = start if start > 0 else 0
        end = self.data_index + 1
        return self.data[start:end]

    


    
    def _reset(self):
        """Return initial_time_step."""
        self.data_index = 0
        self.deposit = self.start_deposit
        self.shares = 0
        return ts.restart(self.get_state())

    
    def _step(self, action: Actions):
        """Apply action and return new time_step."""        

        if self.data_index > len(self.data):
            self.reset()

        if action == Actions.BUY.value:
            shares = buy_shares(self.deposit, self.get_price())
            self.deposit -= shares * self.get_price()
            self.shares += shares
        
        if action == Actions.SELL.value:
            self.deposit += sell_shares(self.shares, self.get_price())
            self.shares = 0

        if self.data_index == len(self.data) - 1:
            self.deposit += sell_shares(self.shares, self.get_price())
            reward = np.log(self.deposit)
            return ts.termination(self.get_state(), reward)
        else:
            self.data_index += 1
            return ts.transition(self.get_state(), reward=0.0, discount=1.0)




if __name__ == '__main__':

    
    data = np.array([[1, 2, 0, 2, 100], [2, 3, 1, 1, 200]], dtype=np.float32)
    env = SimpleTradingEnv(data)  
    

    utils.validate_py_environment(env)

            

        
        




    
