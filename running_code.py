import gym
import torch 
from gym.envs.registration import register
from gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv

from spinup import ddpg_pytorch

register(
    id='gym_cart_pole',
    entry_point='gym.envs.classic_control.continuous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=200,
)


def ddpg_cart_pole_runner(df):

    seed_value = 0

    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001

    discount_factor = df

    env_fn = lambda: gym.make('gym_cart_pole')

    ac_kwargs = dict(hidden_sizes=[400,300], activation=torch.nn.ReLU)

    logger_kwargs = dict(output_dir='./outputs/', exp_name='experiment_name')

    ddpg_pytorch(env_fn,  ac_kwargs=ac_kwargs, seed=seed_value, steps_per_epoch=200, epochs=100, replay_size=1000000, gamma=discount_factor, polyak=0.995, pi_lr=actor_learning_rate, q_lr=critic_learning_rate, batch_size=100, start_steps=10000, update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, max_ep_len=1000, save_freq=1, logger_kwargs=logger_kwargs )


dfs = [0.99, 0.9, 0.7, 0.5]


#for df in dfs:




ddpg_cart_pole_runner(0.99)





