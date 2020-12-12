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
from torch.optim import Adam
import copy
from torch.autograd import Variable
import torch.nn.functional as F
from itertools import count, permutations, product
from sklearn.utils import shuffle
from copy import deepcopy as copy

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
        self.use_cuda = torch.cuda.is_available()
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = Actor(self.state_dim_actor, self.action_dim)
        self.critic = Critic(self.state_dim_critic, self.action_dim, num_agent_lim)
        
        self.target_actor = copy(self.actor)
        self.target_critic = copy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.lr_c) 
        
        ''' Setup CUDA Environment'''
        self.device = 'cuda' if self.use_cuda else 'cpu'
        if self.use_cuda:
            self.actor.to(self.device)
            self.target_actor.to(self.device)
            self.critic.to(self.device)
            self.target_critic.to(self.device)
        
        
            utils.hard_update(self.target_actor, self.actor)
            utils.hard_update(self.target_critic, self.critic)
        self.memories = ReplayBuffer(mem_size)
            
            
    def set_environment(self, env):
        self.env = env
        self.num_agents = env.num_agents
        
    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
    """
        state = Variable(torch.from_numpy(state).to(self.device))
        action = self.target_actor.forward(state).detach()
        return action.to('cpu').data.numpy().argmax()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).to(self.device))
        action = torch.argmax(self.actor.forward(state).detach())
        return int(action.to('cpu').data.numpy())

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s, a, r, ns = self.memories.sample(self.batch_size)      
        reward_predict = []
        pre_acts = []
        for j in range(len(ns)):
            state = ns[j]
            action = self.select_action_from_state(state)
            action = np.array(self.form_action_predict(action), dtype=np.float32)
            pre_acts.append(action)
            state = np.array([state])
            action = np.array([action])
            _state = torch.from_numpy(state).to(self.device)
            action = torch.from_numpy(action).to(self.device)
            reward = self.target_critic(_state, action).to('cpu').data.numpy()[0]
            reward_predict.append(reward[0])
            
        print(r[0], reward_predict[0])
            
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
        y_predicted = torch.squeeze(self.critic.forward(s, a))
        # print(y_predicted)
        # print(y_expected)
        ''' compute critic loss, and update the critic '''
        
        
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
    
        # ---------------------- optimize actor ----------------------[]
        loss_actor = -1*torch.sum(self.critic.forward(s, pre_acts))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
    
        utils.soft_update(self.target_actor, self.actor, self.tau)
        utils.soft_update(self.target_critic, self.critic, self.tau)
        
        self.actor_loss_value = loss_actor.to('cpu').data.numpy()
        self.critic_loss_value = loss_critic.to('cpu').data.numpy()
        self.iter += 1
        
    def get_agent_state(self, agents_pos, agent):
        agent_state = []
        for i in range(20):
            agent_state.append([0] * 20)
        
        x, y = agents_pos[agent]
        agent_state[x][y] = 1
        return agent_state
                
        
    def select_action(self, state, epsilon):
        actions = []
        state = copy(state)
        state = np.reshape(flatten(state), (7, 20, 20))
        state = [state[0], [state[1], state[2]], [state[3], state[4]], state[5], state[6]]
        agent_pos_1 = copy(self.env.agent_pos_1)
        agent_pos_2 = copy(self.env.agent_pos_2)
        
        order = shuffle(range(self.num_agents))
        for i in range(self.num_agents):
            agent = order[i]
            act = None
            
            if random() <= epsilon:
                act = randint(0, self.action_lim - 1)
            else:
                _state = state
                agent_state = self.get_agent_state(agent_pos_1, agent)
                _state = flatten([_state, agent_state])
                act = self.get_exploration_action(np.array(_state, dtype=np.float32))
            valid, state, agent_pos_1, score = self.env.fit_action(i, state, act, agent_pos_1, agent_pos_2, False)
            actions.append(act)
            
        self.steps_done += 1
        return actions
                      
        
    def select_action_smart(self, state):
        actions = [0] * self.num_agents
        state = copy(state)
        state = np.reshape(flatten(state), (7, 20, 20))
        state = [state[0], [state[1], state[2]], [state[3], state[4]], state[5], state[6]]
        agent_pos_1 = copy(self.env.agent_pos_1)
        agent_pos_2 = copy(self.env.agent_pos_2)
        init_score = self.env.score_mine - self.env.score_opponent
        rewards = []
        states = []
        next_states = []
        order = shuffle(range(self.num_agents))
        for i in range(self.num_agents):
            agent = order[i]
            _state = state
            agent_state = self.get_agent_state(agent_pos_1, agent)
            _state = flatten([_state, agent_state])
            states.append(state)
            act = 0
            scores = [0] * 9
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(9):
                _state, _agent_pos_1, _agent_pos_2 = copy([state, agent_pos_1, agent_pos_2])
                valid, _state, _agent_pos, _score = self.env.fit_action(agent, _state, act, _agent_pos_1, _agent_pos_2)
                scores[act] = _score - init_score
                mn = min(mn, _score - init_score)
                mx = max(mx, _score - init_score)
                valid_states.append(valid)
            scores[0] = mn
            for j in range(len(scores)):
                scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
            sum = np.sum(scores) + 0.0001
            for j in range(len(scores)):
                scores[j] = scores[j] / sum
                if(valid_states[j] is False):
                    scores[j] = 0
            act = np.array(scores).argmax()
            valid, state, agent_pos, score = self.env.fit_action(agent, state, act, agent_pos_1, agent_pos_2)
            rewards.append(score - init_score)
            init_score = score
            actions[agent] = act
            next_states.append(state)
            
        return actions
    
    def select_action_test_not_predict(self, state):
        actions = []
        state = copy(state)
        state = np.reshape(flatten(state), (7, 20, 20))
        state = [state[0], [state[1], state[2]], [state[3], state[4]], state[5], state[6]]
        agent_pos_1 = copy(self.env.agent_pos_1)
        agent_pos_2 = copy(self.env.agent_pos_2)
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
                _state, _agent_pos_1, _agent_pos_2 = copy([state, agent_pos_1, agent_pos_2])
                valid, _state, _agent_pos, _score = self.env.fit_action(i, _state, act, _agent_pos_1, _agent_pos_2, False)
                scores[act] = _score - init_score
                mn = min(mn, _score - init_score)
                mx = max(mx, _score - init_score)
                valid_states.append(valid)
            for j in range(len(scores)):
                scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
                scores[j] **= 5
            sum = np.sum(scores) + 0.0001
            for j in range(len(scores)):
                scores[j] = scores[j] / sum
                if(valid_states[j] is False):
                    scores[j] = 0
            act = choices(range(9), scores)[0]
            valid, state, agent_pos, score = self.env.fit_action(i, state, act, agent_pos_1, agent_pos_2)
            init_score = score
            actions.append(act)
            next_states.append(state)
            
        return states, actions, rewards, next_states
    
    def select_best_actions(self, state):
        actions = [0] * self.num_agents
        state = copy(state)
        state = np.reshape(flatten(state), (7, 20, 20))
        state = [state[0], [state[1], state[2]], [state[3], state[4]], state[5], state[6]]
        agent_pos_1 = copy(self.env.agent_pos_1)
        agent_pos_2 = copy(self.env.agent_pos_2)
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
                _state, _agent_pos_1, _agent_pos_2 = copy([state, agent_pos_1, agent_pos_2])
                valid, _state, _agent_pos, _score = self.env.fit_action(agent, _state, act, _agent_pos_1, _agent_pos_2)
                scores[act] = _score - init_score
                mn = min(mn, _score - init_score)
                mx = max(mx, _score - init_score)
                valid_states.append(valid)
            # scores[0] -= 2
            for j in range(len(scores)):
                scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
                scores[j] **= 10
            sum = np.sum(scores) + 0.0001
            for j in range(len(scores)):
                scores[j] = scores[j] / sum
                if(valid_states[j] is False):
                    scores[j] = 0
            scores[0] = 0
            act = choices(range(9), scores)[0]
            valid, state, agent_pos, score = self.env.fit_action(agent, state, act, agent_pos_1, agent_pos_2)
            rewards.append(score - init_score)
            init_score = score
            actions[agent] = act
            next_states.append(state)
            
        return states, actions, rewards, next_states
    
        
    
    def select_random(self, state):
        actions = []
        for i in range(self.num_agents):
            actions.append(randint(0, 8))
        return state, actions, [0] * self.num_agents, state 
        
    def select_action_from_state(self, state):
        actions = []
        state = copy(state)
        state = np.reshape(flatten(state), (7, 20, 20))
        state = [state[0], [state[1], state[2]], [state[3], state[4]], state[5], state[6]]
        agent_pos_1 = copy(self.env.agent_pos_1)
        agent_pos_2 = copy(self.env.agent_pos_2)
        
        order = shuffle(range(self.num_agents))
        for i in range(self.num_agents):
            agent = order[i]
            act = None
            _state = state
            agent_state = self.get_agent_state(agent_pos_1, agent)
            _state = flatten([_state, agent_state])
            act = self.get_exploitation_action(np.array(_state, dtype=np.float32))
            valid, state, agent_pos_1, score = self.env.fit_action(i, state, act, agent_pos_1, agent_pos_2, False)
            actions.append(act)
            
        self.steps_done += 1
        return actions
    
    def transform_to_critic_state(self, state):
        state[1] = self.get_state_critic(state[1])
        return state
    
    def get_state_actor(self):
        return copy([self.env.score_matrix, self.env.agents_matrix, 
                self.env.conquer_matrix, self.env.treasures_matrix, self.env.walls_matrix])
              
    def get_state_critic(self, state = None):
        if state is None:
            state = [self.score_matrix, self.agents_matrix,
                              self.conquer_matrix, self.treasures_matrix]
        state = copy(state)
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
    
    def form_action_predict(self, actions):
        form_actions = []
        for i in range(self.num_agent_lim):
            act = -1
            if (i < len(actions)):
                act = actions[i]
            form_actions.append([1 if i == act else 0 for i in range(9)])
        return flatten(form_actions)
    
    def action_flatten(self, acts):
        _acts = []
        for act in acts:
            p = [1 if j == act else 0 for j in range(self.action_lim)]
            _acts.append(p)
        while(len(_acts) < self.num_agent_lim):
            _acts.append([0] * self.action_lim)
        return flatten(_acts)

    def learn(self, state, actions_1, actions_2, BGame, show_screen):
        next_state, reward, done, remaining_turns = self.env.next_frame(
            actions_1, actions_2, BGame, show_screen)
        
        action = self.form_action_predict(actions_1)
        state = flatten(state)
        next_state = flatten(next_state)
        self.memories.store_transition(state, action, reward, next_state)
            
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
        torch.save(self.target_actor.state_dict(), './Models/target_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/target_critic.pt')
        torch.save(self.actor.state_dict(), './Models/actor.pt')
        torch.save(self.critic.state_dict(), './Models/critic.pt')
        print('Models saved successfully')
        
    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.target_actor.load_state_dict(
            torch.load('./Models/target_actor.pt', map_location = self.device))
        self.target_critic.load_state_dict(
            torch.load('./Models/target_critic.pt', map_location = self.device))
        self.actor.load_state_dict(
            torch.load('./Models/actor.pt', map_location = self.device))
        self.critic.load_state_dict(
            torch.load('./Models/critic.pt', map_location = self.device))
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.actor.eval()
        self.critic.eval()
                
        
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')

