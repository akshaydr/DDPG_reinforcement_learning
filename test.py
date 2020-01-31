import gym
import  numpy as np
import tensorflow as tf
from actorCritic import ActorNetwork

import sys
import os

from replayBuffer import ReplayBuffer
from utilities import build_summaries

def test(sess, env, args, actor, actor_noise):
    # Load ckpt file
    loader = tf.compat.v1.train.Saver()    

    if args['ckpts_file'] is not None:
        ckpt = args['ckpts_dir'] + '/' + args['ckpts_file']  
    else:
        ckpt = tf.train.latest_checkpoint(args['ckpts_dir'])
    
    loader.restore(sess, ckpt)
    sys.stdout.write('%s restored.\n\n' % ckpt)
    sys.stdout.flush() 

    ckpt_split = ckpt.split('-')
    train_ep = ckpt_split[-1]
    
    # Setup Summary
    summary_ops, summary_vars = build_summaries()

    # sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(int(args['max_episodes'])):
        s = env.reset()
        ep_reward = 0

        for j in range(int(args['max_episode_len'])):
            if args['render_env']:
                env.render()

            a = actor.predict(np.reshape(s, (1, actor.s_dim)))
            s2, r, terminal, _ = env.step(a[0])

            s = s2
            ep_reward += r

            if terminal:
                if (summary_ops != None):
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                    })

                    writer.add_summary(summary_str, i)
                    writer.flush()
                
                print('| Reward: {:d} | Episode: {:d}'.format(int(ep_reward), \
                        i))

                break
            