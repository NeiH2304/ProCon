import copy
from math import sqrt, acos, pi

class Environment(object):

    def __init__(self, height = 0, width = 0, score_matrix = [[]], agent_coord_1 = [[]],
                 agent_coord_2 = [[]], coord_treasures = [], coord_walls = [], turns = 0, conquer_matrix = [[], []]):
        self.width = width
        self.height = height
        self.score_matrix = score_matrix
        self.agent_coord_1 = agent_coord_1
        self.agent_coord_2 = agent_coord_2
        self.coord_treasures = coord_treasures
        self.coord_walls = coord_walls
        self.num_actions = 9
        self.num_agents = len(self.agent_coord_1)
        self.turns = turns
        self.remaining_turns = turns
        self.actions = [i for i in range(self.num_actions)]
        self.conquer_matrix = conquer_matrix
        self.agents_matrix = []
        self.treasures_matrix = []
        self.player_1 = 1
        self.player_2 = -1
        self.treasure_score_1 = 0
        self.treasure_score_2 = 0
        self.score_mine = 0
        self.score_opponent = 0
        self.max_scr = -1000
        self.min_scr = 1000
        self.init()
        self.preprocess()
        self.max_scr = -1000
        self.min_scr = 1000
        self.data = [agent_coord_1, agent_coord_2, coord_treasures, turns]
        self.precoord = copy.deepcopy(agent_coord_1)
            
    def init(self):
            
        for coord_wall in self.coord_walls:
            self.score_matrix[coord_wall[0]][coord_wall[1]] = -1000
        
        
    def reset(self):
        
        self.agent_coord_1, self.agent_coord_2,\
                self.coord_treasures, self.turns = copy.deepcopy(self.data)
        self.remaining_turns = self.turns
        self.agents_matrix = []
        self.treasures_matrix = []
        self.conquer_matrix = [[], []]
        self.treasure_score_1 = 0
        self.treasure_score_2 = 0
        self.preprocess()
        
    def preprocess(self):
        if self.turns == 0:
            return
        
        load = (len(self.conquer_matrix[0]) > 0)
        
        for i in range(20):
            arr1 = [0] * 20
            arr2 = [0] * 20
            arr3 = [0] * 20
            arr4 = [0] * 20
                
            self.agents_matrix.append(arr1)
            self.treasures_matrix.append(arr2)
            if not load:
                self.conquer_matrix[0].append(arr3)
                self.conquer_matrix[1].append(arr4)
            for j in range(20):
                if(self.score_matrix[i][j] > -100):
                    self.max_scr = max(self.max_scr, self.score_matrix[i][j])
                    self.min_scr = min(self.min_scr, self.score_matrix[i][j])
        for i in range(self.num_agents):    
            x, y = self.agent_coord_1[i]
            self.agents_matrix[x][y] = i + 1
            if not load:
                self.conquer_matrix[0][x][y] = 1
            x, y = self.agent_coord_2[i]
            self.agents_matrix[x][y] = - i - 1
            if not load:
                self.conquer_matrix[1][x][y] = 1
            
        for coord in self.coord_treasures:
            self.treasures_matrix[coord[0]][coord[1]] = coord[2]
    
    def get_act(act):
        switcher = {
                (0, 0): 0,
                (1, 0): 1,
                (1, 1): 2,
                (0, 1): 3,
                (-1, 1): 4,
                (-1, 0): 5,
                (-1, -1): 6,
                (0, -1): 7,
                (1, -1): 8,
            }
        return switcher.get(act, "nothing")
    
    def compute_score_area(self, state, player):
        area_matrix = []
        score_matrix, agent_matrix, conquer_matrix, treasures_matrix = state
        visit = []
        score = 0
        for i in range(20):
            visit.append([0] * 20)
            area_matrix.append([0] * 20)
            for j in range(20):
                visit[i][j] = conquer_matrix[player][i][j]
            
        def is_border(x, y):
            return x <= 0 or x >= self.height - 1 or y <= 0 or y >= self.width - 1
        
        def can_move(x, y):
            return x >= 0 and x < self.height and y >= 0 and y < self.width \
                and conquer_matrix[player][x][y] != 1
        
        def dfs(x, y):
            visit[x][y] = 1
            area_matrix[x][y] = 1
            temp_score = abs(score_matrix[x][y])
            if(score_matrix[x][y] < -100):
                area_matrix[x][y] = 0
                temp_score = 0
            if is_border(x, y):
                area_matrix[x][y] = 0
                return -1
            dx = [1, -1, 0, 0]
            dy = [0, 0, -1, 1]
            ok = True
            for i in range(4):
                if can_move(x + dx[i], y + dy[i]) and visit[x + dx[i]][y + dy[i]] == 0:
                   _score = dfs(x + dx[i], y + dy[i])
                   if _score == -1:
                       ok = False
                   else:
                       temp_score += _score
            if ok == False:
                area_matrix[x][y] = 0
                return -1
            return temp_score
        
        
        for i in range(20):
            for j in range(20):
                if visit[i][j] == 0:
                    temp = dfs(i, j)
                    score += max(0, temp)
                    
        return score, area_matrix
        
    def compute_score(self, state):
        score_matrix, agent_matrix, conquer_matrix, treasures_matrix = state
        score_title_A = 0
        score_title_B = 0
        treasure_score_1 = 0
        treasure_score_2 = 0
        for i in range(20):
            for j in range(20):
                if(conquer_matrix[0][i][j] == 1):
                    score_title_A += score_matrix[i][j]
                if(conquer_matrix[1][i][j] == 1):
                    score_title_B += score_matrix[i][j]
                if(treasures_matrix[i][j] > 0):
                    if(conquer_matrix[0][i][j] == 1):
                        treasure_score_1 += treasures_matrix[i][j]
                        treasures_matrix[i][j] = 0
                    if(conquer_matrix[1][i][j] == 1):
                        treasure_score_2 += treasures_matrix[i][j]
                        treasures_matrix[i][j] = 0
        score_area_A, area_matrix_1 = self.compute_score_area(state, 0)
        score_area_B, area_matrix_2 = self.compute_score_area(state, 1)
            
        score_A = score_title_A + score_area_A
        score_B = score_title_B + score_area_B
        return score_A, score_B, treasure_score_1, treasure_score_2, area_matrix_1
    
    
    def check_next_action(self, _act, id_agent, agent_coord):
        x, y = agent_coord[id_agent][0], agent_coord[id_agent][1]
        x, y = self.next_action(x, y, _act)
        if not (x >= 0 and x < self.height and y >= 0 and y < self.width):
            return False
        
        return self.score_matrix[x][y] > -100
    
    def next_action(self, x, y, act):
        def action(x):
            switcher = {
                0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1],
                4: [-1, 1], 5: [-1, 0], 6: [-1, -1], 7: [0, -1], 8: [1, -1]
            }
            return switcher.get(x, [0, 0])
        _action = action(act)
        return [x + _action[0], y + _action[1]]
    
    def get_agent_state(self, agent_matrix, agent_id):
        agent_matrix = copy.deepcopy(agent_matrix)
        for i in range(20):
            for j in range(20):
                if agent_matrix[i][j] == agent_id + 1:
                    agent_matrix[i][j] = 1
                else:
                    agent_matrix[i][j] = 0
        
        return agent_matrix
    
    def angle(self, a1, b1, a2, b2):
        fi = acos((a1 * a2 + b1 * b2) / (sqrt(a1*a1 + b1*b1) * (sqrt(a2*a2 + b2*b2))))
        return fi
    
    def check(self, x0, y0, x, y, act):
        
        def action(x):
            switcher = {
                0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1],
                4: [-1, 1], 5: [-1, 0], 6: [-1, -1], 7: [0, -1], 8: [1, -1]
            }
            return switcher.get(x, [0, 0])
        
        a1, b1 = action(act)
        a2, b2 = x - x0, y - y0
        if abs(self.angle(a1, b1, a2, b2)) - 0.0001 <= pi / 3:
            return True
        return False
        
        
    
    def predict_scores(self, x, y, state, predict, act, area_matrix):
        score_matrix, agents_matrix, conquer_matrix, treasures_matrix = state
        score = 0
        discount = 0.02
        reduce_negative = 0.02
        p_1 = 1.3
        p_2 = 1
        for i in range(1, min(8, self.remaining_turns)):
            for j in range(max(0, x - i), min(self.height, x + i + 1)):
                # if j < 0 or j > self.height or k < 0 or k > self.width:
                #     continue
                new_x = j
                new_y = y - i
                if new_y >= 0:
                    if score_matrix[new_x][new_y] > -100: 
                        _sc = treasures_matrix[new_x][new_y] ** p_1
                        if(conquer_matrix[0][new_x][new_y] != 1):
                            _sc += (max(reduce_negative * score_matrix[new_x][new_y], score_matrix[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            if area_matrix[new_x][new_y] == 0:
                                score += _sc * discount
                new_x = j
                new_y = y + i
                if new_y  < self.width:
                    if score_matrix[new_x][new_y] > -100: 
                        _sc = treasures_matrix[new_x][new_y] ** p_1
                        if(conquer_matrix[0][new_x][new_y] != 1):
                            _sc += (max(reduce_negative * score_matrix[new_x][new_y], score_matrix[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            if area_matrix[new_x][new_y] == 0:
                                score += _sc * discount
            for k in range(max(0, y - i), min(self.height, y + i + 1)):
                new_x = x - i
                new_y = k
                if new_x >= 0:
                    if score_matrix[new_x][new_y] > -100: 
                        _sc = treasures_matrix[new_x][new_y] ** p_1
                        if(conquer_matrix[0][new_x][new_y] != 1):
                            _sc += (max(reduce_negative * score_matrix[new_x][new_y], score_matrix[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            if area_matrix[new_x][new_y] == 0:
                                score += _sc * discount
                new_x = x + i
                new_y = k
                if new_x < self.height:
                    if score_matrix[new_x][new_y] > -100: 
                        _sc = treasures_matrix[new_x][new_y] ** p_1
                        if(conquer_matrix[0][new_x][new_y] != 1):
                            _sc += (max(reduce_negative * score_matrix[new_x][new_y], score_matrix[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            if area_matrix[new_x][new_y] == 0:
                                score += _sc * discount
            discount *= 0.7
        # print(score)
        return score
    
    def fit_action(self, agent_id, state, act, agent_coord_1, agent_coord_2, predict = True):
        score_matrix, agents_matrix, conquer_matrix, treasures_matrix = copy.deepcopy(state)
        x, y = agent_coord_1[agent_id][0], agent_coord_1[agent_id][1]     
        new_coord = (self.next_action(x, y, act))
        _x, _y = new_coord
        aux_score = 0
        valid = True
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and score_matrix[_x][_y] > -100:
            # aux_score += (treasures_matrix[_x][_y] + score_matrix[_x][_y]) * 0.4
            if agents_matrix[_x][_y] == 0:
                if conquer_matrix[1][_x][_y] == 0:
                    agents_matrix[_x][_y] = agent_id + 1
                    agents_matrix[x][y] = 0
                    conquer_matrix[0][_x][_y] = 1
                    agent_coord_1[agent_id][0] = _x
                    agent_coord_1[agent_id][1] = _y
                    aux_score += 1
                else:
                    conquer_matrix[1][_x][_y] = 0
                    aux_score -= 0.5
                    # if _x == self.precoord[agent_id][0] and _y == self.precoord[agent_id][1]:
                    #     aux_score -= 2
        else:
            valid = False
        state = [score_matrix, agents_matrix, conquer_matrix, treasures_matrix]
        score_1, score_2, treasures_score_1, treasures_score_2, area_matrix = self.compute_score(state)
        if valid:
            aux_score += self.predict_scores(_x, _y, state, predict, act, area_matrix)
        # print(aux_score, score_1 + treasures_score_1 - score_2 - treasures_score_2)
        if(predict is False):
            aux_score = 0
        self.precoord = copy.deepcopy(self.agent_coord_1)
        return valid, state, agent_coord_1, score_1 + treasures_score_1 - score_2 - treasures_score_2 + aux_score
    
    def next_frame(self, actions_1, actions_2, BGame, change):
        
        point_punish = 30 
        punish = 0
        new_coord_A = []
        new_coord_B = []
        check_A = [0] * self.num_agents
        check_B = [0] * self.num_agents
        
        for i in range(self.num_agents):
            x, y = self.agent_coord_1[i][0], self.agent_coord_1[i][1]
            new_coord_A.append(self.next_action(x, y, actions_1[i]))
            x, y = self.agent_coord_2[i][0], self.agent_coord_2[i][1]
            new_coord_B.append(self.next_action(x, y, actions_2[i]))
            
        for i in range(self.num_agents):
            x, y = new_coord_A[i]
            if(not (x >= 0 and x < self.height and y >= 0 and y < self.width)):
                # print(change, self.agent_coord_1[i][0], self.agent_coord_1[i][1], x, y, ' Warning1!\n')
                check_A[i] = 1
                new_coord_A[i] = [self.agent_coord_1[i][0], self.agent_coord_1[i][1]]
                punish += point_punish
            elif(self.score_matrix[x][y] < -100):
                check_A[i] = 1
                new_coord_A[i] = [self.agent_coord_1[i][0], self.agent_coord_1[i][1]]
                punish += point_punish
            
        for i in range(self.num_agents):
            x, y = new_coord_B[i]
            if(not (x >= 0 and x < self.height and y >= 0 and y < self.width)):
                check_B[i] = 1
                new_coord_B[i] = [self.agent_coord_2[i][0], self.agent_coord_2[i][1]]
            elif(self.score_matrix[x][y] < -100):
                check_B[i] = 1
                new_coord_B[i] = [self.agent_coord_2[i][0], self.agent_coord_2[i][1]]
                    
        # create connect matrix
        connect_matrix = []
        for j in range(2 * self.num_agents):
            connect_matrix.append([0] * (2 * self.num_agents))
            
        for i in range(2 * self.num_agents):
            if i < self.num_agents:
                X = new_coord_A[i]
            else:
                X = new_coord_B[i % self.num_agents]
            for j in range(2 * self.num_agents):
                if i == j:
                    continue
                if j < self.num_agents:
                    Y = self.agent_coord_1[j]
                    if X[0] == Y[0] and X[1] == Y[1]:
                        connect_matrix[i][j] = 1
                else:
                    Y = self.agent_coord_2[j % self.num_agents]
                    if X[0] == Y[0] and X[1] == Y[1]:
                        connect_matrix[i][j] = 1
                        
        # if conflict action to 1 square
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if new_coord_A[i][0] == new_coord_B[j][0] and\
                    new_coord_A[i][1] == new_coord_B[j][1]:
                    check_A[i] = 1
                    check_B[j] = 1
                    
                if new_coord_A[i][0] == new_coord_A[j][0] and\
                    new_coord_A[i][1] == new_coord_A[j][1] and i != j:
                    check_A[i] = 1
                    check_A[j] = 1
                    punish += point_punish
                    
                if new_coord_B[i][0] == new_coord_B[j][0] and\
                    new_coord_B[i][1] == new_coord_B[j][1] and i != j:
                    check_B[i] = 1
                    check_B[j] = 1
        
        
        for i in range(self.num_agents):
            if check_A[i] == 1:
                new_coord_A[i] = [self.agent_coord_1[i][0], self.agent_coord_1[i][1]]
            if check_B[i] == 1:
                new_coord_B[i] = [self.agent_coord_2[i][0], self.agent_coord_2[i][1]]
                    
        # find the clique
        for i in range(2 * self.num_agents):
            if i < self.num_agents:
                if check_A[i] == True:
                    continue
            elif check_B[i - self.num_agents] == True:
                continue
            u = i
            stk = []
            stk.append(u)
            visit = [0] * (2 * self.num_agents)
            visit[u] = 1
            
            for _ in range(2 * self.num_agents):
                for j in range(2 * self.num_agents):
                    if connect_matrix[u][j] == 1 and u != j:
                        stk.append(j)
                        ck = True
                        if j < self.num_agents:
                            if check_A[j] == 1:
                                ck = False
                        else:
                            if check_B[j % self.num_agents] == 1:
                                ck= False
                        if visit[j] == 1:
                            ck = False
                            
                        if not ck:
                            for id in stk:
                                if id < self.num_agents:
                                    check_A[id] = 1
                                else:
                                    check_B[id % self.num_agents] = 1
                            stk = []
                            break
                        u = j
                        visit[j] = 1
        
        # find the remove action
        for i in range(2 * self.num_agents):
            u = i
            stk = []
            visit = [0] * (2 * self.num_agents)
            visit[u] = 1
            if i < self.num_agents:
                if check_A[i] == True:
                    continue
            elif check_B[i % self.num_agents] == True:
                continue
            
            for _ in range(2 * self.num_agents):
                for j in range(2 * self.num_agents):
                    if connect_matrix[u][j] == 1 and u != j:
                        ck = True
                        if j < self.num_agents:
                            x, y = new_coord_A[j]
                            if self.conquer_matrix[1][x][y] == 1:
                                ck = False
                        else:
                            x, y = new_coord_B[j % self.num_agents]
                            if self.conquer_matrix[0][x][y] == 1:
                                ck = False
                        if visit[j] == 1:
                            ck = False
                            stk.append(u)
                        visit[j] = 1
                        
                        if not ck:
                            for id in stk:
                                if id < self.num_agents:
                                    check_A[id] = 1
                                else:
                                    check_B[id % self.num_agents] = 1
                            stk = []
                            break
                        stk.append(j)
                        u = j
                if len(stk) == 0:
                    break
            stk.append(u)
                                
        for i in range(self.num_agents):
            if check_A[i] == 1:
                new_coord_A[i] = [self.agent_coord_1[i][0], self.agent_coord_1[i][1]]
            if check_B[i] == 1:
                new_coord_B[i] = [self.agent_coord_2[i][0], self.agent_coord_2[i][1]]
        
        
        # change before action
        for i in range(self.num_agents):
            if check_A[i] == 0:
                x, y = new_coord_A[i]
                if(self.conquer_matrix[1][x][y] == 0):
                    if self.agent_coord_1[i][0] != new_coord_A[i][0] or self.agent_coord_1[i][1] != new_coord_A[i][1]:
                        self.agents_matrix[self.agent_coord_1[i][0]][self.agent_coord_1[i][1]] = 0
                        self.agents_matrix[x][y] = 0
                        if(change):
                            BGame.redraw_squares(self.agent_coord_1[i][0], self.agent_coord_1[i][1], i + 1)
                else:
                    self.agents_matrix[x][y] = 0
                                      
            if check_B[i] == 0:
                x, y = new_coord_B[i]
                if(self.conquer_matrix[0][x][y] == 0):
                    if self.agent_coord_2[i][0] != new_coord_B[i][0] or self.agent_coord_2[i][1] != new_coord_B[i][1]:
                        self.agents_matrix[x][y] = 0
                        self.agents_matrix[self.agent_coord_2[i][0]][self.agent_coord_2[i][1]] = 0
                        if(change):
                            BGame.redraw_squares(self.agent_coord_2[i][0], self.agent_coord_2[i][1], -i - 1)
                else:
                    self.agents_matrix[x][y] = 0
                        
        # change after action
        for i in range(self.num_agents):
            if check_A[i] == 0:
                x, y = new_coord_A[i]
                if(self.conquer_matrix[1][x][y] == 1):
                    self.conquer_matrix[1][x][y] = 0
                    if(change):
                        BGame.reset_square(x, y, 0)
                    new_coord_A[i] = [self.agent_coord_1[i][0], self.agent_coord_1[i][1]]
                    check_A[i] = 1
                else:
                    self.conquer_matrix[0][x][y] = 1
                    self.agents_matrix[x][y] = i + 1
                    
            if check_B[i] == 0:
                x, y = new_coord_B[i]
                if(self.conquer_matrix[0][x][y] == 1):
                    self.conquer_matrix[0][x][y] = 0
                    if(change):
                        BGame.reset_square(x, y, 0)
                    new_coord_B[i] = [self.agent_coord_2[i][0], self.agent_coord_2[i][1]]
                    check_B[i] = 1
                else:
                    self.conquer_matrix[1][x][y] = 1
                    self.agents_matrix[x][y] = -i - 1
        
        old_score = self.score_mine
                
        for i in range(self.num_agents):
            self.agent_coord_1[i] = [new_coord_A[i][0], new_coord_A[i][1]]
            self.agent_coord_2[i] = [new_coord_B[i][0], new_coord_B[i][1]]
            
        state = [self.score_matrix, self.agents_matrix, self.conquer_matrix, self.treasures_matrix]
        score_A, score_B, treasure_score_1, treasure_score_2, area_matrix = self.compute_score(state)
        self.treasure_score_1 += treasure_score_1
        self.treasure_score_2 += treasure_score_2
        
        if(change):
            for i in range(self.num_agents):
                BGame.reset_square(self.agent_coord_1[i][0], self.agent_coord_1[i][1], i + 1)
                BGame.reset_square(self.agent_coord_2[i][0], self.agent_coord_2[i][1], - i - 1)
            BGame.show_score()
        
        self.score_mine = score_A + self.treasure_score_1
        self.score_opponent = score_B + self.treasure_score_2
        
        reward = self.score_mine - old_score - punish
        self.remaining_turns -= 1
        
        if(change):
            BGame.save_score(self.score_mine, self.score_opponent, self.remaining_turns)
        
        terminal = (self.remaining_turns == 0)
            
        
        return [state, reward, terminal, self.turns - self.remaining_turns]

        
