import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp

import sys 
import os

from actorCritic import ActorNetwork, CriticNetwork
from utilities import OrnsteinUhlenbeckActionNoise
from ddpg import train
from test import  test

def main(args):

    with tf.compat.v1.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.compat.v1.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        
        try:
        # Ensure action bound is symmetric
            assert (env.action_space.high == -env.action_space.low)
        except:
            ValueError

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['train']:
            print("Training Mode")
            train(sess, env, args, actor, critic, actor_noise)
        else:
            print("Testing Mode")
            test(sess, env, args, actor, actor_noise)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.01)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # MountainCarContinuous-v0
    # Ant-v2
    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Ant-v2')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1997)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=600)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')

    parser.add_argument('--ckpts-dir', type=str, help='directory for saving checkpoints', default='./ckpts')
    parser.add_argument('--ckpts_step', help='directory for saving checkpoints')

    # MountainCarContinuous-v0.ckpt-119
    # Ant-v2.ckpt-1300
    parser.add_argument('--load_ckpts', help='Choose whether to train or test the system', action='store_true')
    parser.add_argument('--ckpts_file', type= str, help='directory for loading checkpoints', default='MountainCarContinuous-v0.ckpt-119')

    parser.add_argument('--train', help='Choose whether to train or test the system', action='store_true')

    parser.set_defaults(ckpts_step=200)

    parser.set_defaults(train=True)
    parser.set_defaults(load_ckpts=False)
    parser.set_defaults(render_env=False)
    
    args = vars(parser.parse_args())

    pp.pprint(args)
    main(args)