from __future__ import division
import random
from copy import deepcopy as dcopy

from src.environment import Environment
from src.agents import Agent
from read_input import Data, read_state
from GameBoard.game_board import BoardGame
import pygame
from itertools import count
from collections import deque
import copy
from src.utils import flatten, vizualize
import time

def _test(opt): 
    
    n_games = opt.n_games
    n_maps = opt.n_maps
    data = Data.Read_Input(n_maps)
    BGame = BoardGame(opt.show_screen)
    input_dim_actor = 2000
    input_dim_critic = 2000
    max_agents = 8
    max_actions = 9
    num_agents = len(data[0][3])
    epsilon = opt.initial_epsilon
    
    agent_1 = Agent(opt.gamma, opt.lr_actor, opt.lr_critic, input_dim_actor,
                    input_dim_critic, num_agents, max_agents, max_actions,
                    opt.replay_memory_size, opt.batch_size, 'agent_procon_1',
                        opt.load_checkpoint, opt.saved_path, Environment())
    agent_2 = Agent(opt.gamma, opt.lr_actor, opt.lr_critic, input_dim_actor,
                    input_dim_critic, num_agents, max_agents, max_actions,
                    opt.replay_memory_size, opt.batch_size, 'agent_procon_2',
                    opt.load_checkpoint, opt.saved_path, Environment())
    # if opt.load_checkpoint:
    #     agent_1.load_models(0)
    #     agent_2.load_models(0)
    for _ep in range(opt.n_epochs):
        '''' Read input state '''
        map_id = random.randint(0, n_maps - 1) 
        # map_id = 3
        _map = dcopy(data[map_id])
        print('Testing -- map_id: {}'.format(map_id))
        h, w, score_matrix, coord_agens_1, coord_agens_2,\
        coord_treasures, coord_walls, turns = [infor for infor in _map]
        inp = read_state(opt.file_name)
        h, w, score_matrix, coord_treasures, coord_walls, coord_agens_1, coord_agens_2,\
            conquer_matrix, turns, num_agents = inp
        coord_agens_1 = [x[1:] for x in coord_agens_1]
        coord_agens_2 = [x[1:] for x in coord_agens_2]
        agent_1.set_environment(Environment(h, w, score_matrix, coord_agens_1,
                      coord_agens_2, coord_treasures, coord_walls, turns))
        agent_2.set_environment(Environment(h, w, score_matrix, coord_agens_2,
                      coord_agens_1, coord_treasures, coord_walls, turns))
        
        if opt.show_screen:
            _state_1 = agent_1.get_state_actor()
            BGame.create_board(h, w, _state_1[0], _state_1[1], _state_1[2][0], _state_1[3])
        for _game in range(n_games):
            scores_of_team_1 = deque(maxlen=1000)
            scores_of_team_2 = deque(maxlen=1000)
            start = time.time()
            agent_1.env.reset()
            agent_2.env.reset()
            done = False
            for _iter in count(): 
                epsilon *= opt.discount
                state_1 = agent_1.get_state_actor()
                states_1, actions_1, rewards_1, next_states_1 =\
                    agent_1.select_action_smart(state_1)
                
                state_2 = agent_2.get_state_actor()
                states_2, actions_2, rewards_2, next_states_2 = \
                    agent_2.select_best_actions(state_2)
                # actions_2 = [0] * agent_2.num_agents
                done = agent_1.update_state(
                    states_1, actions_1, rewards_1, next_states_1, actions_2, BGame, opt.show_screen)
                done = agent_2.update_state(
                    states_2, actions_2, rewards_2, next_states_2, actions_1, BGame, False)
                if opt.show_screen:
                    pygame.display.update()
                scores_of_team_1.append(agent_1.env.score_mine)
                scores_of_team_2.append(agent_1.env.score_opponent)
                if done:
                    break
            vizualize(scores_of_team_1, 'Loss_critic_value', 'red')
            vizualize(scores_of_team_2, 'Loss_actor_value', 'blue')
            end = time.time()
            if opt.show_screen:
                BGame.restart()
            print("Time: {}". format(end - start))
            print("Score A/B: {}/{}". format(agent_1.env.score_mine, agent_1.env.score_opponent))
            
        print('Completed episodes')

