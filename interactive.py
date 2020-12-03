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
    coord_agens_A = []
    coord_agens_B = []
    for i in range(len(coord_agens_1)):
        coord_agens_A.append(coord_agens_1[i][1:])
        coord_agens_B.append(coord_agens_2[i][1:])
    agent = Agent(0, 1, 1, 1, 1, num_agents, max_agents, max_actions,\
                  1, 1, 'agent_procon_1', False, None, Environment())
    agent.set_environment(Environment(h, w, score_matrix, coord_agens_A,
                  coord_agens_B, coord_treasures, coord_walls, turns, conquer_matrix))
    state = agent.get_state_actor()
    if opt.type == 1:
        states, actions, rewards, next_states = agent.select_action_smart(state)
    else:
        states, actions, rewards, next_states = agent.select_action_test_not_predict(state)

    def action(x):
        switcher = {
            0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1],
            4: [-1, 1], 5: [-1, 0], 6: [-1, -1], 7: [0, -1], 8: [1, -1]
        }
        return switcher.get(x, [0, 0])
    for i in range(len(actions)):
        actions[i] = [coord_agens_1[i][0], action(actions[i])]
    print(actions)