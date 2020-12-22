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
from random import random, randint, choices, uniform
from src import utils
from src.utils import flatten
from torch.optim import Adam, SGD
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.utils import shuffle
from copy import deepcopy as copy

class Agent():
    def __init__(self, gamma, lr_a, lr_c, state_dim_actor, state_dim_critic, num_agents, num_agent_lim, action_dim,
                 mem_size, batch_size, agent_name, chkpoint, chkpt_dir, env = None):
        
        self.state_dim_actor = state_dim_actor
        self.state_dim_critic = state_dim_critic
        self.action_dim = action_dim
        self.action_lim = action_dim
        self.agent_dim = 400
        self.iter = 0
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.tau = 0.01
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
        
        
        ''' Setup CUDA Environment'''
        self.device = 'cuda' if self.use_cuda else 'cpu'
        
        if chkpoint:
            self.load_models()
        else:
            self.actor = Actor(self.state_dim_actor, self.agent_dim, self.action_dim)
            self.critic = Critic(self.state_dim_critic, self.agent_dim, self.action_dim, num_agent_lim)
            
            self.target_actor = copy(self.actor)
            self.target_critic = copy(self.critic)
            
            if self.use_cuda:
                self.actor.to(self.device)
                self.target_actor.to(self.device)
                self.critic.to(self.device)
                self.target_critic.to(self.device)
                utils.hard_update(self.target_actor, self.actor)
                utils.hard_update(self.target_critic, self.critic)
                
        self.actor_optimizer = Adam(self.actor.parameters(), self.lr_a)
        self.critic_optimizer = Adam(self.critic.parameters(), self.lr_c) 
        
        self.memories = ReplayBuffer(mem_size)
            
            
    def set_environment(self, env):
        self.env = env
        self.num_agents = env.num_agents
        
    def get_exploitation_action(self, state, agent_state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
    """
        state = Variable(torch.from_numpy(state).to(self.device))
        agent_state = Variable(torch.from_numpy(agent_state).to(self.device))
        action = self.target_actor.forward(state, agent_state).detach()
        return action.to('cpu').data.numpy()[0]

    def get_exploration_action(self, state, agent_state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).to(self.device))
        agent_state = Variable(torch.from_numpy(agent_state).to(self.device))
        action = self.actor.forward(state, agent_state).detach()
        return action.to('cpu').data.numpy()[0]

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        if self.memories.len < self.batch_size:
            return
        
        s, pos, a, r, ns, npos = self.memories.sample(self.batch_size)    
        
        ags = torch.tensor([], requires_grad=True)
        nags = torch.tensor([], requires_grad=True)
        _a = torch.tensor([], requires_grad=True)
        for i in range(len(s)):
            action_1 = torch.tensor([], requires_grad=True)
            action_2 = torch.tensor([], requires_grad=True)
            action_3 = torch.tensor([], requires_grad=True)
            for agent in range(self.num_agent_lim):
                _ags = flatten(self.get_agent_state(pos[i][agent]))
                _nags = flatten(self.get_agent_state(npos[i][agent]))
                action_3 = torch.cat((action_3, self.actor.forward(
                    torch.from_numpy(np.array([flatten(s[i])], dtype = np.float32)).to(self.device),
                    torch.from_numpy(np.array([flatten(_ags)], dtype = np.float32)).to(self.device))))
                
                action_1 = torch.cat((action_1, self.target_actor.forward(
                    torch.from_numpy(np.array([flatten(s[i])], dtype = np.float32)).to(self.device),
                    torch.from_numpy(np.array([flatten(_ags)], dtype = np.float32)).to(self.device))))
                action_2 = torch.cat((action_2, self.target_actor.forward(
                    torch.from_numpy(np.array([flatten(ns[i])], dtype = np.float32)).to(self.device),
                    torch.from_numpy(np.array([flatten(_nags)], dtype = np.float32)).to(self.device))))
            ags = torch.cat((ags, action_1))
            nags = torch.cat((nags, action_2))
            _a = torch.cat((_a, action_3))
        s = Variable(torch.from_numpy(s).to(self.device), requires_grad=True)
        ags = Variable(ags, requires_grad=True)
        a = Variable(torch.from_numpy(a).to(self.device), requires_grad=True)
        r = Variable(torch.from_numpy(r).to(self.device), requires_grad=True)
        ns = Variable(torch.from_numpy(ns).to(self.device), requires_grad=True)
        # _a = Variable(_a)
        # nags = Variable(nags, requires_grad=True)
        ''' ---------------------- optimize ----------------------
        Use target actor exploitation policy here for loss evaluation
        y_exp = r + gamma*Q'( s2, pi'(s2))
        y_pred = Q( s1, a1)
        '''
        next_val = torch.squeeze(self.target_critic.forward(ns, nags).detach())
        y_expected = r + self.gamma * next_val
        y_predicted = 0 + torch.squeeze(self.critic.forward(s, ags))
        ''' compute critic loss, and update the critic '''
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        ''' ---------------------- optimize actor ----------------------'''
        loss_actor = -1 * torch.sum(self.critic.forward(s, _a))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        # for parameter in self.actor.parameters():
        #     if(parameter.grad is None):
        #         print(parameter)
            # print(parameter.grad)
        utils.soft_update(self.target_actor, self.actor, self.tau)
        utils.soft_update(self.target_critic, self.critic, self.tau)
        
        self.actor_loss_value = -loss_actor.to('cpu').data.numpy()
        self.critic_loss_value = loss_critic.to('cpu').data.numpy()
        
    def get_agent_state(self, pos):
        agent_state = []
        for i in range(20):
            agent_state.append([0] * 20)
        x, y = pos
        x = int(x)
        y = int(y)
        if x >= 0 and y >= 0:
            agent_state[x][y] = 1
        return agent_state
                
        
    def select_action(self, state, epsilon):
        actions = [[]] * self.num_agents
        order = shuffle(range(self.num_agents))
        agent_pos_1 = copy(self.env.agent_pos_1)
        
        for i in range(self.num_agents):
            agent = order[i]
            action = []
            agent_state = self.get_agent_state(agent_pos_1[agent])
            agent_state = np.array([flatten(agent_state)], dtype=np.float32)
            if random() <= epsilon:
                action = [0] * self.action_dim
                action = np.array(action, dtype = np.float32)
                S = 0
                for j in range(len(action)):
                    action[j] = randint(0, 1000)
                    S += action[j]
                for j in range(len(action)):
                    action[j] = action[j] * 1.0 / S
            else:
                # print(agent_state)
                action = self.get_exploration_action(np.array([flatten(state)], dtype=np.float32), agent_state)
            actions[agent] = action
            # print(action)
        
        # print(self.num_agents)
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
            actions[agent] = scores
            next_states.append(state)
            
        return actions[0]
    
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
        act = self.get_exploration_action(np.array(flatten(state), dtype=np.float32))
        return [act]
    
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

    def learn(self, state, actions_1, actions_2, BGame, show_screen, saved):
        # act = [actions_1.argmax()]
        _actions_1 = copy(actions_1)
        _actions_2 = copy(actions_2)
        acts_1 = []
        acts_2 = []
        for agent in range(self.num_agents):
            for i in range(9):
                _actions_1[agent][i] = _actions_1[agent][i] ** 3
                _actions_2[agent][i] = _actions_2[agent][i] ** 3
            act_1 = choices([i for i in range(9)], _actions_1[agent]) 
            act_2 = choices([i for i in range(9)], _actions_2[agent]) 
            acts_1.append(act_1[0])
            acts_2.append(act_2[0])
        
        pos = copy(self.env.agent_pos_1)
        
        next_state, reward, done, remaining_turns = self.env.next_frame(
            acts_1, acts_2, BGame, show_screen)
        
        if not saved:
            return
        
        if acts_1[0] == 0:
            reward -= 3
        next_pos = copy(self.env.agent_pos_1)
        
        state = flatten(state)
        next_state = flatten(next_state)
        for i in range(self.num_agent_lim - self.num_agents):
            pos.append([-1, -1])
            next_pos.append([-1, -1])
        self.memories.store_transition(state, pos, actions_1, reward, next_state, next_pos)
            
        self.optimize()

        return done

    def update_state(self, states_1, actions_1, rewards_1, next_states_1, actions_2, BGame, show_screen):
        next_state, reward, done, remaining_turns = self.env.next_frame(
            actions_1, actions_2, BGame, show_screen)
        return done

    def save_models(self):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:        
        """
        torch.save(self.target_actor, './Models/target_actor.pt')
        torch.save(self.target_critic, './Models/target_critic.pt')
        torch.save(self.actor, './Models/actor.pt')
        torch.save(self.critic, './Models/critic.pt')
        print('Models saved successfully')
        
    def load_models(self):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.target_actor = torch.load('./Models/target_actor.pt', map_location = self.device)
        self.target_critic = torch.load('./Models/target_critic.pt', map_location = self.device)
        self.actor = torch.load('./Models/actor.pt', map_location = self.device)
        self.critic = torch.load('./Models/critic.pt', map_location = self.device)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.actor.eval()
        self.critic.eval()
                
        
        # utils.hard_update(self.target_actor, self.actor)
        # utils.hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')

