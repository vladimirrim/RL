from gym import make
import numpy as np
import torch
from torch import nn
from copy import deepcopy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.linear_model import SGDRegressor
from torch.autograd import Variable
import copy
from collections import deque
import random
import math
import pickle
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

N_STEP = 3
GAMMA = 0.96
wandb.init('dul')
SEED = 8888



class Featurizer:
    def __init__(self):
        env = make("MountainCar-v0")
        env.seed(SEED)
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        self.featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100, random_state=SEED)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100, random_state=SEED)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100, random_state=SEED)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100, random_state=SEED))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def transform_state(self, state):
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return [featurized[0]]

class Agent:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.step = 0
        with open('agent.pkl', 'rb') as f:
          self.models, self.featurizer = pickle.load(f)
        #self.featurizer = Featurizer()
        #self.models = []
        #for _ in range(action_dim):
        #    model = SGDRegressor(random_state=SEED)
        #    model.partial_fit(self.featurizer.transform_state(env.reset()), [0])
        #    self.models.append(model)
        self.target = deepcopy(self.models)

    def update(self, transition):
        state, action, next_state, reward, done = transition
        state = self.featurizer.transform_state(state)
        next_state = self.featurizer.transform_state(next_state)
        self.models[action].partial_fit(state, [self.gamma * int(done) * 
                                                [t.predict(next_state)[0] for t in self.target][np.argmax([t.predict(next_state)[0] for t in self.models])]
                                                + reward])


    def act(self, state, target=False):
        self.step += 1
        state = self.featurizer.transform_state(state)
        if (self.step % 500) == 0:
          self.target = deepcopy(self.models)
        return np.argmax([t.predict(state)[0] for t in self.models])

    def save(self):
        with open('agent.pkl', 'wb') as f:
          pickle.dump((self.models, self.featurizer), f)


if __name__ == "__main__":
    env = make("MountainCar-v0")
    aql = Agent(state_dim=2, action_dim=3)
    eps = 0.1
    episodes = 2000

    for i in range(episodes):
        state = env.reset()
        vel = state
        total_reward = 0
        steps = 0
        done = False
        eps *= 0.99
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = aql.act(state)
            next_state, reward, done, _ = env.step(action)
            next_vel = next_state
            reward += 300 * (GAMMA * abs(next_vel[1]) - abs(vel[1]))
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                aql.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
            vel = next_vel
        wandb.log({'reward': steps})
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                aql.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        if i % 20 == 0:
            aql.save()