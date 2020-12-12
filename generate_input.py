import random


for i in range(2000):
    file_name = 'Input_File/inp_file_' + str(i) + '.txt'
    f = open(file_name, 'w')
    m = random.randint(20, 20)
    n = random.randint(20, 20)
        
    score_matrix = []
    mx = random.randint(3, 30)
    matrix = []
    for i in range(m):
        matrix.append([0] * n)
        score_matrix.append([0] * n)
        
    for i in range(m):
        for j in range(n):
            value = random.randint(-mx, mx)
            if(value < 0 and random.randint(0, 1) == 0):
                value = -value
            score_matrix[i][j] =  value
            score_matrix[m - i - 1][n - j - 1] = value
    
    turn = random.randint(30, 60)
    
    num_agens = random.randint(2, 8)
    
    coord = [0] * (num_agens * 2)
    
    
    for j in range(num_agens):
        _x, _y = random.randint(0, m - 1), random.randint(0, n - 1)
        while  _x == _y or matrix[_x][_y] > 0: 
            _x = random.randint(0, m - 1)
            _y = random.randint(0, n - 1)
        matrix[_x][_y] = 1
        matrix[m - _x - 1][n - _y - 1] = 2
        coord[j] = [_x, _y]
        coord[j + num_agens] = [m - _x - 1, n - _y - 1]
    
        
    num_treasures = random.randint(3, 10)
    
    treasure_coord = [0] * (num_treasures * 2)
    for j in range(num_treasures):
        _x, _y = random.randint(0, m - 1), random.randint(0, n - 1)
        while  _x == _y or matrix[_x][_y] > 0: 
            _x = random.randint(0, m - 1)
            _y = random.randint(0, n - 1)
        # score_matrix[_x][_y] = random.randint(8, 16)
        matrix[_x][_y] = 3
        matrix[m - _x - 1][n - _y - 1] = 3
        # score_matrix[m - _x - 1][n - _y - 1] = random.randint(8, 16)
        value = random.randint(8, 16)
        treasure_coord[j] = [_x, _y, value]
        treasure_coord[j + num_treasures] = [m - _x - 1, n - _y - 1, value]
    
           
    num_walls = random.randint(int(m * n / 30), int(m * n / 20))
    
    wall_coord = [0] * (num_walls * 2)
    for j in range(num_walls):
        _x, _y = random.randint(0, m - 1), random.randint(0, n - 1)
        while  _x == _y or matrix[_x][_y] > 0: 
            _x = random.randint(0, m - 1)
            _y = random.randint(0, n - 1)
        matrix[_x][_y] = 4
        matrix[m - _x - 1][n - _y - 1] = 4
        wall_coord[j] = [_x, _y]
        wall_coord[j + num_walls] = [m - _x - 1, n - _y - 1]
    

    s = str(m) + ' ' + str(n) + '\n'
    f.write(s)
    for i in range(m):
        for j in range(n):
            f.write(str(score_matrix[i][j]) + ' ')
        f.write('\n')
    
    f.write(str(turn) + '\n')
    f.write(str(num_agens) + '\n')
    for _pair in coord:
        f.write(str(_pair[0]) + ' ' + str(_pair[1]) + '\n')
    
    f.write(str(num_treasures * 2) + '\n')
    for _pair in treasure_coord:
        f.write(str(_pair[0]) + ' ' + str(_pair[1]) + ' ' + str(_pair[2]) + '\n')
    f.write(str(num_walls * 2) + '\n')
    for _pair in wall_coord:
        f.write(str(_pair[0]) + ' ' + str(_pair[1]) + '\n')
    