"""
This script shows how to adapt an environment to be compatible
with the OpenAI Gym-style API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Getting Started with Gym" section of the OpenAI 
gym documentation.

The following snippet was used to demo basic functionality.

    import gym
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import QRDQN, TQC
from stable_baselines3 import HerReplayBuffer

from stable_baselines3 import A2C

from datetime import datetime
import os

import gym

if __name__ == "__main__":

    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSITION")

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="Sawyer",  # use Sawyer robot
            controller_configs=controller_config,
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )

    #env = gym.make('FetchReach-v1')

    now = datetime.today()
    dt_string = now.strftime("%H_%M_%S")
    print("date and time =", dt_string)	

    filename = 'test_Lift_Panda_TQC' + dt_string

    #TQC issue with her buffer ? obs must be dict ?
    hyp = {  'policy': 'MultiInputPolicy' , 'buffer_size': 1000000, 'ent_coef': 'auto', 'batch_size': 1024, 'gamma': 0.95, 'learning_starts': 1000, 'learning_rate':1e-3, 'replay_buffer_class': HerReplayBuffer, 'replay_buffer_kwargs': {'online_sampling': True, 'goal_selection_strategy': 'future', 'n_sampled_goal': 4, 'max_episode_length' : 50}, 'policy_kwargs': {'net_arch': [512, 512, 512], 'n_critics': 2}}

    #PPO ?
    #hyp = { 'policy': 'MultiInputPolicy' , 'ent_coef': 'auto', 'batch_size': 1024, 'gamma': 0.95, 'learning_rate':1e-3, 'policy_kwargs': {'net_arch': [512, 512, 512]}}

    model = TQC(
        env=env,
        tensorboard_log="./a2c_cartpole_tensorboard12/",
        seed=2246026888,
        verbose=True,
        **hyp,
        )

    

    model.learn(total_timesteps=2000000, tb_log_name=filename)

    replay_buffer_path = os.path.join('trained_models3', 'replay_buffer' + filename + '.pkl')

    model.save('trained_models3/' + filename)
    env.save('trained_models3/vec_normalize_' + filename + '.pkl')     # Save VecNormalize statistics
    model.save_replay_buffer(replay_buffer_path)

    
    check_env(env)

    for i_episode in range(100):
        observation = env.reset()
        for t in range(10):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print()
            print('action' , action)
            print()
            print('observation' , observation)
            print()
            print('reward' , reward)
            print()
            print('done', done)
            print()

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
