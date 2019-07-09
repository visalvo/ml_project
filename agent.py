import random
from random import randint
from collections import deque
import numpy as np

import config
import q_network


class DDQNAgent:
    def __init__(self, state_size, action_size, train=True):

        # self.accelerate = True
        self.t = 0
        self.max_Q = 0
        self.train = train

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = config.DISCOUNT
        self.learning_rate = config.LEARNING_RATE
        if self.train:
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6

        self.epsilon_min = config.EPSILON_MIN
        self.batch_size = config.BATCH_SIZE
        self.train_start = config.TRAIN_START
        self.explore = config.EXPLORE

        # Create replay memory using deque
        self.memory = deque(maxlen=config.MAX_REPLAY)

        self.model = self.build_model(config.MODEL_TYPE)
        self.target_model = self.build_model(config.MODEL_TYPE)

        # Copy the model to target model
        self.update_target_model()

    def build_model(self, type):
        if type == 1:
            return q_network.atari_model(self.action_size, self.learning_rate, self.state_size)
        elif type == 2:
            return q_network.custom_model(self.action_size, self.learning_rate, self.state_size)
        # elif type == 3:
        #    return q_network.custom_cnn_model(self.action_size, self.learning_rate, self.state_size)
        else:
            raise ValueError('Illegal type model')

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            return randint(0, self.action_size - 1)
        else:
            q_arr = self.model.predict(s_t)
            if not len(q_arr[0]) == self.action_size:
                raise ValueError('Illegal array length, must be ', self.action_size)
            return np.argmax(q_arr[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*mini_batch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)
