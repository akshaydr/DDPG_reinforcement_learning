import gym
import  numpy as np
import tensorflow as tf
from actorCritic import ActorNetwork

def test(sess, env, args, actor, critic, actor_noise):
    env = gym.make(args['env'])
    np.random.seed(int(args['random_seed']))
    tf.compat.v1.set_random_seed(int(args['random_seed']))
    env.seed(int(args['random_seed']))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    
    actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                        float(args['actor_lr']), float(args['tau']),
                        int(args['minibatch_size']))
