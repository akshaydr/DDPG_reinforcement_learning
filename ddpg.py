import tensorflow as tf
import numpy as np

from replayBuffer import ReplayBuffer
from utilities import build_summaries

import os
import sys

import csv

#   Agent Training
def train(sess, env, args, actor, critic, actor_noise):
    # Load ckpt file
    if args['load_ckpts']:
        print("Loading checkpoints")
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
    else:
        print("Starting new training")
        sess.run(tf.compat.v1.global_variables_initializer())
        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()
        train_ep = 0
    

    # Define saver for saving model ckpts
    model_name = str(env) + '.ckpt'
    checkpoint_path = os.path.join(args['ckpts_dir'], model_name)        
    if not os.path.exists(args['ckpts_dir']):
        os.makedirs(args['ckpts_dir'])
    saver = tf.compat.v1.train.Saver() 

    # Setup Summary
    summary_ops, summary_vars = build_summaries()

    # sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    # actor.update_target_network()
    # critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    for i in range(int(train_ep) + 1, int(args['max_episodes']) + 1):

        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Add exploration noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, _ = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                # Find argmax q value of the current episode
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            csv_write = [i, ep_reward, ep_ave_max_q]

            if terminal:
                if (summary_ops != None):
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(j)
                    })

                    writer.add_summary(summary_str, i)
                    writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                break

        if (i % int(args['ckpts_step']) == 0):
            saver.save(sess, checkpoint_path, i)
            sys.stdout.write('Checkpoint saved \n')
            sys.stdout.flush()

        with open('results/rewards.csv', mode='a', newline='') as output_file:
            output_writer = csv.writer(output_file, lineterminator='\n')
            output_writer.writerow(csv_write)