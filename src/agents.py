#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:45:45 2020

@author: hien
"""
import numpy as np
import torch
from src.deep_q_network import Critic, Actor
from src.replay_memory import ReplayBuffer
from random import random, randint, choices
from src import utils
from src.utils import flatten
from copy import deepcopy
from torch.optim import Adam
import copy
from torch.autograd import Variable
import torch.nn.functional as F
from itertools import count, permutations, product
from sklearn.utils import shuffle

class Agent():
    def __init__(self, gamma, lr_a, lr_c, state_dim_actor, state_dim_critic, num_agents, num_agent_lim, action_dim,
                 mem_size, batch_size, agent_name, chkpoint, chkpt_dir, env = None):
        
        self.state_dim_actor = state_dim_actor
        self.state_dim_critic = state_dim_critic
        self.action_dim = action_dim
        self.action_lim = action_dim
        self.iter = 0
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.tau = 0.05
        self.steps_done = 0
        self.nrand_action = 0
        self.gamma = gamma
        self.num_agent_lim = num_agent_lim
        self.max_n_agents = self.num_agent_lim
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir
        self.env = env
        self.critic_loss_value = 0
        self.actor_loss_value = 0
        self.chkpoint = chkpoint
        self.num_agents = num_agents
        self.agent_name = agent_name
        self.use_cuda = False
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actors = [Actor(self.state_dim_actor, self.action_dim) for i in range(num_agent_lim)]
        self.critics = [Critic(self.state_dim_critic,
                               self.action_dim, num_agent_lim) for i in range(num_agent_lim)]
        
        self.target_actors = deepcopy(self.actors)
        self.target_critics = deepcopy(self.critics)
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(), self.lr_a) for i in range(num_agent_lim)]
        self.critic_optimizers = [torch.optim.Adam(self.critics[i].parameters(), self.lr_c) for i in range(num_agent_lim)]
        
        ''' Setup CUDA Environment'''
        self.device = 'cuda' if self.use_cuda else 'cpu'
        if self.use_cuda:
            for i in range(num_agent_lim):
                self.actors[i].to(self.device)
                self.target_actors[i].to(self.device)
                self.critics[i].to(self.device)
                self.target_critics[i].to(self.device)
        
        
        for i in range(num_agent_lim):
            utils.hard_update(self.target_actors[i], self.actors[i])
            utils.hard_update(self.target_critics[i], self.critics[i])
        self.memories = [ReplayBuffer(mem_size) for i in range(num_agent_lim)]
            
            
    def set_environment(self, env):
        self.env = env
        self.num_agents = env.num_agents
        
    def get_exploitation_action(self, state, agent):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
		"""
        state = Variable(torch.from_numpy(state).to(self.device))
        action = self.target_actors[agent].forward(state).detach()
        return action.to('cpu').data.numpy().argmax()

    def get_exploration_action(self, state, agent):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).to(self.device))
        action = self.actors[agent].forward(state).detach()
        return action.to('cpu').data.numpy().argmax()

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        for i in range(self.num_agents):
        
            s, a, r, ns = self.memories[i].sample(self.batch_size)      
            reward_predict = []
            pre_acts = []
            for j in range(len(ns)):
                state = ns[j]
                reward = torch.max(self.target_actors[i](torch.from_numpy(
                    np.array([state])).to(self.device))).to('cpu').data.numpy()
                # print(reward)
                reward_predict.append(reward)
                
            reward_predict = np.array(reward_predict)
            pre_acts = np.array(pre_acts, dtype=np.float32)
            
            s = Variable(torch.from_numpy(s).to(self.device))
            a = Variable(torch.from_numpy(a).to(self.device))
            r = Variable(torch.from_numpy(r).to(self.device))
            ns = Variable(torch.from_numpy(ns).to(self.device))
            pre_acts = Variable(torch.from_numpy(pre_acts).to(self.device))
            reward_predict = torch.squeeze(torch.from_numpy(reward_predict).to(self.device))
            
            ''' ---------------------- optimize ----------------------
            Use target actor exploitation policy here for loss evaluation
            y_exp = r + gamma*Q'( s2, pi'(s2))
            y_pred = Q( s1, a1)
            '''
            y_expected = r + self.gamma * reward_predict
            y_predicted = torch.squeeze(self.actors[i].forward(s))
            # print(y_predicted)
            
            y_predicted = torch.amax(y_predicted, dim = -1)
            # print(y_expected)
            # print(y_predicted)
            ''' compute critic loss, and update the critic '''
            loss_actor = F.mse_loss(y_predicted, y_expected)
            # print(y_expected)
            self.actor_optimizers[i].zero_grad()
            loss_actor.backward()
            self.actor_optimizers[i].step()
            
            utils.soft_update(self.target_actors[i], self.actors[i], self.tau)
            self.actor_loss_value = loss_actor.to('cpu').data.numpy()
            # for param in self.actors[i].parameters():
            #     print(param.data)
        self.iter += 1
        
    def select_action(self, state, epsilon):
        actions = []
        state = copy.deepcopy(state)
        state = np.reshape(flatten(state), (5, 20, 20))
        state = [state[0], state[1], [state[2], state[3]], state[4]]
        agent_coord_1 = copy.deepcopy(self.env.agent_coord_1)
        agent_coord_2 = copy.deepcopy(self.env.agent_coord_2)
        init_score = self.env.score_mine - self.env.score_opponent
        rewards = []
        states = []
        next_states = []
        
        for i in range(self.num_agents):
            act = None
            states.append(state)
            
            if random() <= epsilon:
                act = randint(0, self.action_lim - 1)
            else:
                _state = state
                _state[1] = self.env.get_agent_state(_state[1], i)
                _state = flatten(_state)
                act = self.get_exploration_action(np.array(_state, dtype=np.float32), i)
                
            # _state, _agent_coord_1, _agent_coord_2 = copy.deepcopy([state, agent_coord_1, agent_coord_2])
            valid, state, agent_coord_1, score = self.env.fit_action(i, state, act, agent_coord_1, agent_coord_2, False)
            punish = 0
            if valid is False:
                punish = 20
            rewards.append(score - init_score - punish)
            # rewards.append(score - init_score)
            actions.append(act)
            next_states.append(state)
            init_score = score
            
        self.steps_done += 1
        return states, actions, rewards, next_states
                      
        
    def select_action_smart(self, state):
        actions = [0] * self.num_agents
        state = copy.deepcopy(state)
        state = np.reshape(flatten(state), (5, 20, 20))
        state = [state[0], state[1], [state[2], state[3]], state[4]]
        agent_coord_1 = copy.deepcopy(self.env.agent_coord_1)
        agent_coord_2 = copy.deepcopy(self.env.agent_coord_2)
        init_score = self.env.score_mine - self.env.score_opponent
        rewards = []
        states = []
        next_states = []
        order = shuffle(range(self.num_agents))
        for i in range(self.num_agents):
            agent = order[i]
            _state = state
            _state[1] = self.env.get_agent_state(_state[1], agent)
            _state = flatten(_state)
            states.append(state)
            act = 0
            scores = [0] * 9
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(9):
                _state, _agent_coord_1, _agent_coord_2 = copy.deepcopy([state, agent_coord_1, agent_coord_2])
                valid, _state, _agent_coord, _score = self.env.fit_action(agent, _state, act, _agent_coord_1, _agent_coord_2)
                scores[act] = _score - init_score
                mn = min(mn, _score - init_score)
                mx = max(mx, _score - init_score)
                valid_states.append(valid)
            # scores[0] -= 2
            for j in range(len(scores)):
                scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
                scores[j] **= 4
            sum = np.sum(scores) + 0.0001
            for j in range(len(scores)):
                scores[j] = scores[j] / sum
                if(valid_states[j] is False):
                    scores[j] = 0
            scores[0] = 0
            act = choices(range(9), scores)[0]
            valid, state, agent_coord, score = self.env.fit_action(agent, state, act, agent_coord_1, agent_coord_2)
            rewards.append(score - init_score)
            init_score = score
            actions[agent] = act
            next_states.append(state)
            
        return states, actions, rewards, next_states
    
    def select_action_test_not_predict(self, state):
        actions = []
        state = copy.deepcopy(state)
        state = np.reshape(flatten(state), (5, 20, 20))
        state = [state[0], state[1], [state[2], state[3]], state[4]]
        agent_coord_1 = copy.deepcopy(self.env.agent_coord_1)
        agent_coord_2 = copy.deepcopy(self.env.agent_coord_2)
        init_score = self.env.score_mine - self.env.score_opponent
        rewards = []
        states = []
        next_states = []
        
        for i in range(self.num_agents):
            _state = state
            _state[1] = self.env.get_agent_state(_state[1], i)
            _state = flatten(_state)
            states.append(state)
            act = 0
            scores = [0] * 9
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(9):
                _state, _agent_coord_1, _agent_coord_2 = copy.deepcopy([state, agent_coord_1, agent_coord_2])
                valid, _state, _agent_coord, _score = self.env.fit_action(i, _state, act, _agent_coord_1, _agent_coord_2, False)
                scores[act] = _score - init_score
                mn = min(mn, _score - init_score)
                mx = max(mx, _score - init_score)
                valid_states.append(valid)
            for j in range(len(scores)):
                scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
                scores[j] **= 3
            sum = np.sum(scores) + 0.0001
            for j in range(len(scores)):
                scores[j] = scores[j] / sum
                if(valid_states[j] is False):
                    scores[j] = 0
            act = choices(range(9), scores)[0]
            valid, state, agent_coord, score = self.env.fit_action(i, state, act, agent_coord_1, agent_coord_2)
            init_score = score
            actions.append(act)
            next_states.append(state)
            
        return states, actions, rewards, next_states
    
    def select_best_actions(self, state):
        actions = []
        state = copy.deepcopy(state)
        state = np.reshape(flatten(state), (5, 20, 20))
        state = [state[0], state[1], [state[2], state[3]], state[4]]
        agent_coord_1 = copy.deepcopy(self.env.agent_coord_1)
        agent_coord_2 = copy.deepcopy(self.env.agent_coord_2)
        init_score = self.env.score_mine - self.env.score_opponent
        rewards = []
        states = []
        next_states = []
        acts = [(i + 1) for i in range(8)]
        all_acts = []
        permutations_object = [list(p) for p in product(acts, repeat=self.num_agents)]
        permutations_list = list(permutations_object)
        all_acts += permutations_list
        
        max_score = 0
        
        for acts in all_acts:
            temp_scores = 0
            for i in range(self.num_agents):
                act = acts[i]
                state, agent_coord, score = self.env.fit_action(i, state, act, agent_coord_1, agent_coord_2)
                rewards.append(score - init_score)
                temp_scores += score-  init_score
                init_score += score
                actions.append(act)
                next_states.append(state)
            if(temp_scores > max_score):
                max_score = temp_scores
                actions = acts
        return states, actions, rewards, states
    
        
    
    def select_random(self, state):
        actions = []
        for i in range(self.num_agents):
            actions.append(randint(0, 8))
        return state, actions, [0] * self.num_agents, state 
        
    def select_action_from_state(self, state):
        actions = []
        state = copy.deepcopy(state)
        state = np.reshape(flatten(state), (5, 20, 20))
        state = [state[0], state[1], [state[2], state[3]], state[4]]
        agent_coord_1 = copy.deepcopy(self.env.agent_coord_1)
        agent_coord_2 = copy.deepcopy(self.env.agent_coord_2)
        init_score = self.env.score_mine
        rewards = []
        states = []
        next_states = []
        
        for i in range(self.num_agents):
            _state = state
            _state[1] = self.env.get_agent_state(_state[1], i)
            _state = flatten(_state)
            act = self.get_exploration_action(np.array(_state, dtype=np.float32), i)
            states.append(state)
    
            state, agent_coord, score = self.env.fit_action(i, state, act, agent_coord_1, agent_coord_2)
            
            rewards.append(score - init_score)
            init_score = score
            actions.append(act)
            next_states.append(state)
            
        return states, actions, rewards, next_states
    
    def transform_to_critic_state(self, state):
        state[1] = self.get_state_critic(state[1])
        return state
    
    def get_state_actor(self):
        return copy.deepcopy([self.env.score_matrix, self.env.agents_matrix, 
                self.env.conquer_matrix, self.env.treasures_matrix])
              
    def get_state_critic(self, state = None):
        if state is None:
            state = [self.score_matrix, self.agents_matrix,
                              self.conquer_matrix, self.treasures_matrix]
        state = copy.deepcopy(state)
        state[1] = self.get_all_agent_matrix(state[1])
        return state
    
    def get_all_agent_matrix(self, agents_matrix):
        all_matrix = []
        for k in range(8):
            matrix = []
            for i in range(20):
                matrix.append([0] * 20)
                for j in range(20):
                    if agents_matrix[i][j] == k:
                        matrix[i][j] = 1
                
            all_matrix.append(matrix)
        return all_matrix
    
    def action_flatten(self, acts):
        _acts = []
        for act in acts:
            p = [1 if j == act else 0 for j in range(self.action_lim)]
            _acts.append(p)
        while(len(_acts) < self.num_agent_lim):
            _acts.append([0] * self.action_lim)
        return flatten(_acts)

    def learn(self, states_1, actions_1, rewards_1, next_states_1, actions_2, BGame, show_screen):
        next_state, reward, done, remaining_turns = self.env.next_frame(
            actions_1, actions_2, BGame, show_screen)
        
        for i in range(self.num_agents):
            states_1[i] = flatten(states_1[i])
            next_states_1[i] = flatten(next_states_1[i])
            self.memories[i].store_transition(states_1[i], actions_1[i], rewards_1[i], next_states_1[i])
            
        self.optimize()

        return done

    def update_state(self, states_1, actions_1, rewards_1, next_states_1, actions_2, BGame, show_screen):
        next_state, reward, done, remaining_turns = self.env.next_frame(
            actions_1, actions_2, BGame, show_screen)
        return done

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        for i in range(self.num_agents):
            torch.save(self.target_actors[i].state_dict(), './Models/' + str(episode_count) + '_target_actor' + str(i) + '.pt')
            torch.save(self.target_critics[i].state_dict(), './Models/' + str(episode_count) + '_target_critic' + str(i) + '.pt')
            torch.save(self.actors[i].state_dict(), './Models/' + str(episode_count) + '_actor' + str(i) + '.pt')
            torch.save(self.critics[i].state_dict(), './Models/' + str(episode_count) + '_critic' + str(i) + '.pt')
        print('Models saved successfully')
        
    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(
                torch.load('./Models/' + str(episode) + '_target_actor' + str(i) + '.pt', map_location = self.device))
            # self.target_critics[i].load_state_dict(
            #     torch.load('./Models/' + str(episode) + '_target_critic' + str(i) + '.pt', map_location = self.device))
            self.actors[i].load_state_dict(
                torch.load('./Models/' + str(episode) + '_actor' + str(i) + '.pt', map_location = self.device))
            # self.critics[i].load_state_dict(
            #     torch.load('./Models/' + str(episode) + '_critic' + str(i) + '.pt', map_location = self.device))
            utils.hard_update(self.target_actors[i], self.actors[i])
            # utils.hard_update(self.target_critics[i], self.critics[i])
        print('Models loaded succesfully')

            