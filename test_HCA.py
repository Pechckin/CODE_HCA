#from MountainCar import Continuous_MountainCarEnv
from SimpEnv import SimpEnv
import gym
import torch
import numpy as np
from HCA import HCA

env = SimpEnv()

hca = HCA(episodes=5,
          trajectory=2,
          alpha_actor=1e-4,
          alpha_credit=1e-4,
          gamma=0.95)
rewards = []
for i in range(hca.episodes):
    print(f'Episode {i + 1} --- ')
    done = False
    state = env.reset()
    total_reward = 0
    while not done:
        #env.render()
        action = hca.act(state)

        next_state, reward, done, _ = env.step(action)
        #print(action, ' - действие', reward, ' - награда')
        total_reward += reward
        hca.memory.push(state, action, next_state, reward)
        state = next_state
        hca.update()
    rewards.append(total_reward)
print(rewards)

