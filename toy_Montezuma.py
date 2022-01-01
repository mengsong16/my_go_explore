from __future__ import print_function
import numpy as np
import copy
import time

# text version Montezuma

# 7 discrete actions
DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
JUMP = 4
JUMP_LEFT = 5
JUMP_RIGHT = 6

ACTION_SIZE = 7

# init env
init_map = [
[' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
['D',' ',' ',' ',' ',' ',' ',' ',' ',' ','D'],
['G','G','G',' ','G','L','G',' ','G','G','G'],
[' ',' ',' ',' ',' ','L',' ',' ','S',' ',' '],
['K',' ',' ',' ',' ','L',' ',' ','S',' ',' '],
['G','L',' ',' ','G','G','G',' ','S',' ','L'],
[' ','L',' ',' ',' ',' ',' ',' ',' ',' ','L'],
['G','G','G','G','G','G','G','G','G','G','G']
]

# only door, key could be changed by agent's action
char2num = {
    ' ':0,
    'A':1,  # agent
    'D':2,  # door
    'G':3,  # ground
    'L':4,  # ladder
    'K':5,  # key 
    'S':6   # string
}

# 11*8
MAP_H = len(init_map)
MAP_W = len(init_map[0])

# game status
WIN = 1
FAIL = -1
NORMAL = 0

FEATURE_SIZE = MAP_H * MAP_W
#MATRIX_FEATURE_SIZE = (MAP_H, MAP_W, 3)

gameStatusText = {
    1: 'Win',
    0: 'Normal',
    -1: 'Fail'
}

# count how many keys or doors have been met
key_cnt = 0
door_cnt = 0

def two_d_to_one_d(x, y):
    return x * MAP_W + y

# from text env to num env
def num_encode(list_map):
    a = np.zeros((MAP_H, MAP_W))
    for x in range(MAP_H):
        for y in range(MAP_W):
            a[x,y] = char2num[list_map[x][y]]
    return a

#------------------------------------------------------
class MontezumaWorld(object):
    def __init__(self):
        self.n_actions = ACTION_SIZE
        self.n_features = FEATURE_SIZE
        # set random seed
        self.random = np.random.RandomState(0)

    # sample a random action
    def random_action(self):
        return self.random.randint(self.n_actions)

    # get initial world state
    def sample_scenario(self):
        return MontezumaScenario((1,5), self)

class MontezumaScenario(object):
    def __init__(self, agent_position, world):
        self.agent_position = agent_position
        self.world = world
        self.base_map = copy.deepcopy(init_map)
        #print('(((((((((((((((((', self.base_map)

    def init(self):
        s = MontezumaState(self.agent_position, self.base_map, 0, False, 0, 0, NORMAL)
        return s


class MontezumaState(object):
    def __init__(self, agent_position, base_map, momentum, has_key, fall_dist, step_cnt, gameStatus):
        self.agent_position = agent_position
        self.base_map = base_map
        self.momentum = momentum
        self.has_key = has_key
        self.fall_dist = fall_dist
        self.step_cnt = step_cnt
        self._cached_features = None
        #self._cached_vector_features = None
        # self._cached_matrix_features = None
        # self._cached_matrix_features_multichannel = None
        #self.terminate = False
        self._gameStatus = gameStatus

    def features(self):
        # if none map is stored, return initial map
        if self._cached_features is None:
            tmp_map = copy.deepcopy(self.base_map)
            tmp_map[self.agent_position[0]][self.agent_position[1]] = 'A'
            encode_map = num_encode(tmp_map)
            # map: a vector (including agent)
            self._cached_features = encode_map.flatten()
        return self._cached_features

    # def vector_features(self):
    #     if self._cached_vector_features is None:
    #         tmp_map = self.base_map
    #         tmp_map[self.agent_position[0]][self.agent_position[1]] = 'A'
    #         encode_map = num_encode(tmp_map)
    #         self._cached_vector_features = encode_map.flatten()
    #     return self._cached_vector_features

    # def matrix_features(self):
    #     if self._cached_matrix_features is None:
    #         a = np.zeros((MAP_W, MAP_H))
    #         for p in self.dot_positions:
    #             a[p[0],p[1]] = 1
    #         for p in self.ghost_positions:
    #             a[p[0],p[1]] = 2
    #         p = self.pacman_position
    #         a[p[0],p[1]] = 3
    #         self._cached_matrix_features = a
    #     return self._cached_matrix_features

    # def matrix_features_multichannel(self):
    #     if self._cached_matrix_features_multichannel is None:
    #         a = np.zeros((MAP_W, MAP_H,3))
    #         for p in self.dot_positions:
    #             a[p[0],p[1],0] = 1
    #         for p in self.ghost_positions:
    #             a[p[0],p[1],1] = 1
    #         p = self.pacman_position
    #         a[p[0],p[1],2] = 1
    #         self._cached_matrix_features_multichannel = a
    #     return self._cached_matrix_features_multichannel

    # def dist_features(self):
    #     def my_dist(x, y):
    #         return (x[0] - y[0], x[1] - y[1] )  # keep the direction info

    #     if self._cached_dist_features is None:
    #         tmp = self.pacman_position
    #         for x in self.dot_positions:
    #             tmp = np.concatenate((tmp, my_dist(x, self.pacman_position) ) )
    #         for x in self.ghost_positions:
    #             tmp = np.concatenate((tmp, my_dist(x, self.pacman_position) ) )
    #         self._cached_dist_features = tmp
    #     return self._cached_dist_features


    # def win(self):
    #     if self.dot_position == self.pacman_position:
    #         self.terminate = True
    #         return True
    #     else:
    #         return False

    # transit one step
    def step(self, action):
        def empty(x, y):
            return (self.base_map[x][y] in [' ', 'K']) or (self.base_map[x][y] == 'D' and self.has_key == True)

        def climbable(x,y):
            return self.base_map[x][y] in ['L', 'S']

        def onGround(x, y):
            # x - vertical, down is large
            # y - horizontal, right is large
            if not (0 <= x < MAP_H and 0 <= y < MAP_W and 0 <= x+1 < MAP_H):
                return False
            return (self.base_map[x+1][y] in ['G', 'D'] and (empty(x,y) or climbable(x,y)) ) \
                   or (climbable(x+1,y) and empty(x,y))

        # p: agent's current position           
        def move(p, action):
            x, y = p[0], p[1]
            # move actions
            # compute new position
            if action == DOWN:
                dx, dy = (1, 0)
            elif action == UP:
                dx, dy = (-1, 0)
            elif action == LEFT:
                dx, dy = (0, -1)
            elif action == RIGHT:
                dx, dy = (0, 1)
            elif action == JUMP:
                dx, dy = (-1, 0)
            elif action == JUMP_LEFT:
                dx, dy = (-1, -1)
            elif action == JUMP_RIGHT:
                dx, dy = (-1, 1)
            #print('-', action)
            nx, ny = x + dx, y + dy

            # return current position as new position if out of map or ...
            if not (0 <= nx < MAP_H and 0 <= ny < MAP_W) or not (empty(nx, ny) or climbable(nx, ny)):
                return (x, y)

            # return current position
            # cannot climb down
            if action == DOWN:
                if not (climbable(nx, ny) or climbable(x,y)):
                    nx, ny = x, y
            # cannot climb up        
            elif action == UP:
                if not climbable(x, y):
                    nx, ny = x, y
            # cannot walk left or right        
            elif action in [LEFT, RIGHT]:
                if not onGround(x, y):
                    nx, ny = x, y
            # cannot jump        
            elif action == JUMP:
                if not onGround(x,y):
                    nx, ny = x, y
            # cannot jump left or jump right        
            elif action in [JUMP_LEFT, JUMP_RIGHT]:
                if not (onGround(x,y) or climbable(x,y)):
                    nx, ny = x, y
            # return new position        
            return (nx, ny)

        # def fall_down(p):
        #     x, y = p[0], p[1]
        #     while (empty(x+1, y)):
        #         x = x+1
        #     return (x, y)

        #print('######################',action)

        # game ends (win or fail)
        if self._gameStatus != NORMAL:
            return self._gameStatus, False, 0, self

        # Game fail after 1000 steps    
        if self.step_cnt > 1000:
            return FAIL, False, 0, self

        #--------------move
        x, y = self.agent_position[0], self.agent_position[1]
        # momentum is used to extend jump to the next frame
        momentum = 0
        has_key = self.has_key
        #  how high the agent falls from
        fall_dist = 0
        # action success: action ends
        if self.momentum != 0:
            nx, ny = x + 1, y + self.momentum
            if not (0 <= nx < MAP_H and 0 <= ny < MAP_W) or not (empty(nx, ny) or climbable(nx, ny)):
                nx, ny = x, y
            action_success = False
        elif not climbable(x,y) and x < MAP_H-1 and empty(x+1, y):
            #print('debug:', x+1, y, empty(x+1, y))
            #print('111111111111111')
            nx, ny = x+1, y
            fall_dist = self.fall_dist + 1
            if onGround(nx, ny) and fall_dist > 1: # dead because fall from high
                return FAIL, False, -1.0, MontezumaState((nx,ny), self.base_map, 0, has_key, fall_dist, self.step_cnt+1, FAIL)
            action_success = False
        else:
            (nx, ny) = move(self.agent_position, action)
            #print('debug:', nx, ny)
            #print('22222222222222')
            action_success = (nx, ny) != self.agent_position
            if action_success:
                if action == JUMP_LEFT:
                    momentum = -1
                elif action == JUMP_RIGHT:
                    momentum = 1

        #---------------modify info
        #print('debug:', self.base_map[nx][ny])
        reward = 0.0
        new_base_map = copy.deepcopy(self.base_map)
        #print('$$$$', new_base_map)

        # change map
        if self.base_map[nx][ny] == 'D':
            new_base_map[nx][ny] = ' '
            # door is opened
            has_key = False
            reward = 1.0
            #print('here0')
        if self.base_map[nx][ny] == 'K':
            new_base_map[nx][ny] = ' '
            has_key = True
            reward = 0.4

        #---------------update game status
        flag = NORMAL
        #print('debug2:', self.base_map[nx][ny])
        if self.base_map[nx][ny] == 'D': # agent on door
            #print('here1')
            flag = WIN

        # global key_cnt
        # global door_cnt

        # if reward > 0.9:
        #     door_cnt += 1
        #     if door_cnt < 300:
        #         print('Win!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     elif door_cnt % 300 == 0:
        #         print('Door Count:', door_cnt)
        # elif reward > 0.1:
        #     key_cnt += 1
        #     if key_cnt < 300:
        #         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Get postive reward:', reward)
        #     elif key_cnt % 300 == 0:
        #         print('Key Count:', key_cnt)

        return flag, action_success, reward, MontezumaState((nx, ny), new_base_map, momentum, has_key, fall_dist, self.step_cnt+1, flag)

    # visualize agent position, map, game status
    def visualize(self):
        a = np.array(self.base_map)
        a[self.agent_position] = 'A'
        for i in range(MAP_H):
            for j in range(MAP_W):
                print(a[i,j], end='')
            print('')
        print('Game Status: '+gameStatusText[self._gameStatus])
