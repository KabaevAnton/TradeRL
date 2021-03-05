import unittest

import numpy as np 
import pandas as pd
from common.downloader import YahooDownloader
from environtment.simple_trading_env import SimpleTradingEnv

from tf_agents.environments import tf_py_environment
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import tensorflow as tf


class AgentTrainTest(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_simple_trading_agent(self) -> None:
        downloader = YahooDownloader('2000-01-01','2020-01-01', ['MSFT'])
        msft_df = downloader.fetch_data()
        msft_df = msft_df[['open', 'high', 'low', 'close', 'volume']]
        trading_env = SimpleTradingEnv(np.array(msft_df.values[:30], dtype=np.float32))
        trading_env = tf_py_environment.TFPyEnvironment(trading_env)

        categorical_q_net = categorical_q_network.CategoricalQNetwork(trading_env.observation_spec(), trading_env.action_spec(), fc_layer_params=(100,))
        optimizer = tf.keras.optimizers.Adam()

        agent = categorical_dqn_agent.CategoricalDqnAgent(
            trading_env.time_step_spec(),
            trading_env.action_spec(),
            categorical_q_network=categorical_q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            
            )

        agent.initialize()
        trading_env.reset()
        buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=trading_env.batch_size, max_length=100000)

        #print(trading_env.deposit)
        while True:
            step = trading_env.current_time_step()            
            action_step = agent.collect_policy.action(step)            
            next_step = trading_env.step(action_step.action)
            traj = trajectory.from_transition(step, action_step, next_step)
            buffer.add_batch(traj)

            if step.reward != 0.0:                
                break
        dataset = buffer.as_dataset(single_deterministic_pass=True, num_steps=2, sample_batch_size=1)
        for d in dataset:
            train_loss = agent.train(d[0])
            print(train_loss.loss)

        #iterator = iter(dataset)
        #experience, _ = next(iterator)
        #train_loss = agent.train(dataset)

        #print('train_loss = {}'.format(train_loss.loss))
        