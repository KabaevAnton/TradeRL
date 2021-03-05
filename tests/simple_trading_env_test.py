import unittest
import numpy as np
import pandas as pd

from environtment.simple_trading_env import SimpleTradingEnv, Actions


class SimpleTradingEnvTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_state(self):
        data = np.array([[1,2,3],[4,5,6],[7,8,9], [10, 11, 12]])
        assert data.shape == (4,3)
        trading_env = SimpleTradingEnv(data)
        state = trading_env.get_state()
        self.assertEqual(np.sum(state), np.sum(np.array([[1,2,3]])))

        trading_env = SimpleTradingEnv(data, history_len=1)
        state = trading_env.get_state()
        self.assertEqual(np.sum(state), np.sum(data[0:1]))
        trading_env.data_index = 3
        state = trading_env.get_state()
        self.assertEqual(np.sum(state), np.sum(data[2:4]))

        trading_env = SimpleTradingEnv(data, history_len=2)
        trading_env.data_index = 2
        state = trading_env.get_state()        
        self.assertEqual(np.sum(state), np.sum(data[0:3]))


    def test_get_price(self):
        data = np.array([['01.01.2001', 1, 2, 0, 2, 100, 'MSFT'], ['02.01.2001', 2, 3, 1, 1, 100, 'MSFT']], dtype=object)
        trading_env = SimpleTradingEnv(data, close_price_idx=4)
        self.assertEqual(trading_env.get_price(), 2)
        trading_env.data_index = 1
        self.assertEqual(trading_env.get_price(), 1)


    def test_environtment_trajectory_strategy(self):
        data = np.array([[10], [11], [8], [5], [9], [12], [10], [15]])
        actions = [0, 1, 0, 2, 2, 1, 0, 2]
        assert data.shape[0] == len(actions)
        trading_env = SimpleTradingEnv(data, deposit=10, close_price_idx=0)
        trading_env.reset()
        #0/10->11/0->3/8->3/8->3/8->15/0->5/10->20/0
        
        trading_env.step(actions[0])
        self.assertEqual(trading_env.deposit, 0)        
            
        trading_env.step(actions[1])
        self.assertEqual(trading_env.deposit, 11)        

        trading_env.step(actions[2])
        self.assertEqual(trading_env.deposit, 3)        

        trading_env.step(actions[3])
        self.assertEqual(trading_env.deposit, 3)        

        trading_env.step(actions[4])
        self.assertEqual(trading_env.deposit, 3)        

        trading_env.step(actions[5])
        self.assertEqual(trading_env.deposit, 15)        

        trading_env.step(actions[6])
        self.assertEqual(trading_env.deposit, 5)        

        step = trading_env.step(actions[7])
        self.assertEqual(trading_env.deposit, 20)        

        


        


    

        

        


