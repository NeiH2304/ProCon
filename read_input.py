
class Data():
    
    score_matrix = []
    
    def Read_Input(num_inputs = 1):
        
        MAX_SIZE = 20
        
        data = []
        
        for i in range(num_inputs):
            file_name = 'Input_File/inp_file_' + str(i) + '.txt'
            with open(file_name) as f:
                score_matrix = []
                h, w = map(int, f.readline().split())
                for i in range(h):
                    array = list(map(int, f.readline().split()))
                    while(len(array) < MAX_SIZE):
                        array.append(-1000)
                    score_matrix.append(array)
                while(len(score_matrix) < MAX_SIZE):
                    score_matrix.append([-1000] * MAX_SIZE)
                
                # print(score_matrix)
                turns = list(map(int, f.readline().split()))[0]
                # print(turns)
                num_agens = list(map(int, f.readline().split()))[0]
                coord_agens_of_team_A = []
                coord_agens_of_team_B = []
                for j in range(num_agens * 2):
                    coord = list(map(int, f.readline().split()))
                    # print(coord)
                    if(j < num_agens):
                        coord_agens_of_team_A.append(coord)
                    else:
                        coord_agens_of_team_B.append(coord)
            
                num_tresures = list(map(int, f.readline().split()))[0]
                treasures = []
                for j in range(num_tresures):
                    coord = list(map(int, f.readline().split()))
                    treasures.append(coord)
            
            
                num_walls = list(map(int, f.readline().split()))[0]
                coord_walls = []
                for j in range(num_walls):
                    coord = list(map(int, f.readline().split()))
                    coord_walls.append(coord)
                data.append([h, w, score_matrix, coord_agens_of_team_A,
                            coord_agens_of_team_B, treasures,
                            coord_walls, turns])
        return data