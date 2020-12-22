from __future__ import division
import random
from copy import deepcopy as dcopy

from src.environment import Environment
from src.agents import Agent
from read_input import Data
from GameBoard.game_board import BoardGame
import pygame
from itertools import count
import torch
import numpy as np
from src.utils import flatten, vizualize
from collections import deque
import time

def train(opt): 
    
    n_games = opt.n_games
    n_maps = opt.n_maps
    data = Data.Read_Input(n_maps)
    BGame = BoardGame(opt.show_screen)
    input_dim_actor = 2800
    input_dim_critic = 2800
    max_agents = 8
    max_actions = 9
    num_agents = len(data[0][3])
    epsilon = opt.initial_epsilon
    
    agent_1 = Agent(opt.gamma, opt.lr_actor, opt.lr_critic, input_dim_actor, input_dim_critic, num_agents, max_agents, max_actions,
                    opt.replay_memory_size, opt.batch_size, 'agent_procon_1', opt.load_checkpoint,
                    opt.saved_path, Environment())
    agent_2 = Agent(opt.gamma, opt.lr_actor, opt.lr_critic, input_dim_actor, input_dim_critic, num_agents, max_agents, max_actions,
                    opt.replay_memory_size, opt.batch_size, 'agent_procon_2', False,
                    opt.saved_path, Environment())

    Loss_critic_value = deque(maxlen=10000)
    Loss_actor_value = deque(maxlen=10000)
    
    for _ep in range(opt.n_epochs):
        '''' Read input state '''
        map_id = random.randint(0, 555)
        _map = dcopy(data[map_id])
        print('Training_epochs: {} -- map_id: {}'.format(_ep + 1, map_id))
        h, w, score_matrix, coord_agens_1, coord_agens_2,\
        coord_treasures, coord_walls, turns = [infor for infor in _map]
        agent_1.set_environment(Environment(h, w, score_matrix, coord_agens_1,
                      coord_agens_2, coord_treasures, coord_walls, turns))
        agent_2.set_environment(Environment(h, w, score_matrix, coord_agens_2,
                      coord_agens_1, coord_treasures, coord_walls, turns))
        num_agents = len(coord_agens_1)
        
        if opt.show_screen:
            _state_1 = agent_1.get_state_actor()
            BGame.create_board(h, w, _state_1)
        for _game in range(n_games):
        
            file_name = 'Output_File/out_file_' + '1' + '.txt'
            f = open(file_name, 'a+')
            agent_1.env.reset()
            agent_2.env.reset()
            done = False
            start = time.time()
            
            for _iter in count(): 
                epsilon *= opt.discount
                state_1 = agent_1.get_state_actor()
                actions_1 = agent_1.select_action(state_1, epsilon)
                
                state_2 = agent_2.get_state_actor()
                actions_2= agent_2.select_action(state_2, epsilon)
                # actions_2 = [0] * agent_2.num_agents
                done = agent_1.learn(state_1, actions_1, actions_2, BGame, opt.show_screen, True)
                done = agent_2.learn(agent_2.get_state_actor(), actions_2, actions_1, BGame, False, False)
                if opt.show_screen:
                    pygame.display.update()
                
                
                Loss_critic_value.append(agent_1.critic_loss_value)
                Loss_actor_value.append(agent_1.actor_loss_value)
                
                if _iter % 3 == 0:
                    for i in range(agent_1.num_agents):
                        _state = agent_1.get_state_actor()
                        agent_state = agent_1.get_agent_state(agent_1.env.agent_pos_1[0])
                        agent_state = torch.from_numpy(np.array([flatten(agent_state)], dtype=np.float32)).to(agent_1.device)
                        _state = np.array([flatten(_state)], dtype=np.float32)
                        a  = agent_1.actor(
                            torch.from_numpy(_state).to(agent_1.device), agent_state).data.to('cpu').numpy()[0]
                        for j in range(len(a)):
                            a[j] = round(a[j], 6)
                        print(a)
                    print('Iteration :- ', _iter, ' Loss_actor :- ', agent_1.actor_loss_value,\
                          ' Loss_critic :- ', agent_1.critic_loss_value)
                # print('Epsilon: {}'. format(epsilon))
                # f.write(str(agent_1.actor_loss_value) + ' ' + str(agent_1.critic_loss_value) + '\n')
                if done:
                    break
            
            vizualize(Loss_critic_value, 'Loss_critic_value', 'red')
            vizualize(Loss_actor_value, 'Loss_actor_value', 'blue')
            if opt.show_screen:
                BGame.restart()
            end = time.time()
            f.close()
            ''' save models '''
            if opt.saved_checkpoint:
                agent_1.save_models()
                # agent_2.save_models()
            print("Time: {}". format(end - start))
            print("Score A/B: {}/{}". format(agent_1.env.score_mine, agent_1.env.score_opponent))
            
        print('Completed episodes')

