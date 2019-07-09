import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

import donkey_gym
import config
import utils
from agent import DDQNAgent
import os


def build_action(act_index, last_throttle):
    '''
    index = (config.ACTION_SPACE / 2) - 1
    if act_index <= index:
        steering = act_index * (2 / index) - 1

        if last_throttle > config.THROTTLE_MIN:
            last_throttle -= 0.1
    else:
        steering = (act_index / index) - 1

        if last_throttle < config.THROTTLE_MAX:
            last_throttle += 0.1
    '''

    steering = (act_index * (2 / (config.ACTION_SPACE - 1))) - 1
    # throttle = config.THROTTLE_MIN

    '''
        if agent.accelerate and throttle <= throttle_max:
            throttle += 0.05

        if not agent.accelerate and throttle >= throttle_min:
            throttle -= 0.05
    '''

    return [steering, last_throttle]


conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
K.set_session(sess)

env = gym.make(config.ENV_NAME)

state_size = (config.IMG_ROWS, config.IMG_COLS, config.IMG_STACK)
# action_size = env.action_space.n  # Steering and Throttle
action_size = config.ACTION_SPACE

agent = DDQNAgent(state_size, action_size, config.TRAIN)

episodes = []

if not agent.train:
    print("loading model...")
    agent.load_model(config.MODEL_PATH + config.MODEL_NAME)

for e in range(config.EPISODES):

    done = False
    obs = env.reset()
    episode_len = 0

    q_values = []
    rewards = []

    throttle = config.THROTTLE_MIN
    x_t = utils.process_image(obs)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # In Keras, need to reshape

    while not done:

        action_index = agent.get_action(s_t)
        action = build_action(action_index, throttle)

        next_obs, reward, done, info = env.step(action)

        x_t1 = utils.process_image(next_obs)
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :config.IMG_STACK-1], axis=3)

        # Save the sample <s, a, r, s'> to the replay memory
        agent.replay_memory(s_t, action_index, reward, s_t1, done)

        if agent.train:
            agent.train_replay()

        s_t = s_t1
        agent.t = agent.t + 1
        episode_len = episode_len + 1

        q_values.append(agent.max_Q)
        rewards.append(reward)

        if agent.t % 30 == 0:
            print("EPISODE", e, "STEP", agent.t, "/ ACTION", action, "ACT_INDEX", action_index, "/ REWARD", reward,
                  "/ EPISODE LENGTH", episode_len, "/ Q_MAX ", agent.max_Q)

        if done:

            # Every episode update the target model to be same with model
            agent.update_target_model()

            avg_q = np.mean(q_values)
            avg_reward = np.mean(rewards)

            # Save model for each episode
            if agent.train:
                if not os.path.exists(config.MODEL_PATH):
                    os.makedirs(config.MODEL_PATH)

                agent.save_model(config.MODEL_PATH + config.MODEL_NAME)
                episodes.append([e, episode_len, avg_q, avg_reward, len(agent.memory), agent.epsilon])

                df = pd.DataFrame(episodes,
                                  columns=["episode", "episode_len", "avg_Q-value", "avg_reward", "memory_len",
                                           "epsilon"])
                df.to_csv(config.STATS_FILE, index=None, header=True)

            print("EPISODE:", e, "  MEMORY LEN:", len(agent.memory), "  EPSILON:", agent.epsilon,
                  " EPISODE LEN:", episode_len, " AVG_Q:", avg_q, " AVG_REWARD:", avg_reward)
