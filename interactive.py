from __future__ import division
from src.environment import Environment
from src.agents import Agent
from read_input import read_state

def interactive(opt): 
    max_agents = 8
    max_actions = 9
    
    inp = read_state(opt.file_name)
    h, w, score_matrix, coord_treasures, coord_walls, coord_agens_1, coord_agens_2,\
        conquer_matrix, turns, num_agents = inp
        
    agent = Agent(opt.gamma, 1, 1, 1, 1, num_agents, max_agents, max_actions,\
                  1, 1, 'agent_procon_1', False, opt.saved_path, Environment())
    agent.set_environment(Environment(h, w, score_matrix, coord_agens_1,
                  coord_agens_2, coord_treasures, coord_walls, turns, conquer_matrix))
    state = agent.get_state_actor()
    states, actions, rewards, next_states = agent.select_action_test(state)

    def action(x):
        switcher = {
            0: [0, 0], 1: [-1, 0], 2: [-1, 1], 3: [0, -1],
            4: [1, 1], 5: [1, 0], 6: [1, -1], 7: [0, -1], 8: [-1, -1]
        }
        return switcher.get(x, [0, 0])
    for i in range(len(actions)):
        actions[i] = [coord_agens_1[0], action(actions[i])]
    print(actions)