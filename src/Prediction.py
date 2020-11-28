#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:25:25 2020

@author: hien
"""
import numpy as np
from random import random, randint

class Prediction():
    def __init__(self, model, env, state, epsilon):
        self.model = model
        self.num_agents = env.num_agents
        self.epsilon = epsilon
        self.state = state
        self.prediction = model(state)
        self.env = env
        self.prediction_list = np.array(self.prediction.tolist())
        
    def set_prediction(self, state):
        self.prediction = self.model(state)
        
    def get_action(self):
        
        action_agents = []
        
        for i in range(self.num_agents):
            act = 0
            random_action = random() <= self.epsilon
            if random_action:
                # print("Perform a random action")
                act = randint(0, 8)
            else:
                # print("Perform the best action")
                x = i * 8
                y = (i + 1) * 8
                act = self.prediction_list[x:y].argmax()
            
            while not self.env.check_next_action(act, i):
                act = randint(0, 8)
            
            action_agents.append(act)
            
        return action_agents