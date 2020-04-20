# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces


class SimpEnv(gym.Env):

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = 0.0
        self.max_position = 10.0

        self.low_state = np.array([self.min_position])
        self.high_state = np.array([self.max_position])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.seed()
        self.reset()


    def step(self, action):
        des = True if action > 0.5 else False
        reward = 1.0 if des else -1.0
        self.state = self.state + 1.0 if not des else np.array([self.max_position])
        done = True if des or self.state == np.array([self.max_position]) else False
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0.0])
        return np.array(self.state)


if __name__ == '__main__':
    pass
    #print(np.array([1]) + 1)
