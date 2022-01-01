from __future__ import print_function
import numpy as np
from toy_Montezuma import MontezumaWorld, FEATURE_SIZE, ACTION_SIZE
import time
from gym.spaces import Discrete, Box

gameStatus = {
    1: 'Win',
    0: 'Normal',
    -1: 'Fail'
}


class MontezumaWrapper(object):
    def _init_game(self):
        self.scenario = self.world.sample_scenario()
        self.state = self.scenario.init()
        self._game_status = 0
        if self.display:
            print('------------- New Game -----------------')
            self.state.visualize()

    def __init__(self, display=False):
        self.display = display
        self.world = MontezumaWorld()
        self._init_game()

    def move_success(self):
        return self._action_success

    def act(self, action):
        self._game_status, self._action_success, reward, self.state = self.state.step(action)
        # if self._game_status != 0:
        #     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        if self.display:
            print()
            print('Action: '+str(action))
            self.state.visualize()
            print('Action Success:', self._action_success, ', Reward:', reward)
            time.sleep(3)

        return reward

    def game_over(self):
        return self._game_status != 0

    # get map    
    def get_feature(self):
        return self.state.features()

    def reset_game(self):
        self._init_game()

class MontezumaWrapper_gym(object):
    def __init__(self, display=False, noLifeReward=False, history=1):
        self.display = display
        self.noLifeReward = noLifeReward
        # state history
        self.history = history  
        # 1D array, each element is a discrete value
        self.action_space = Discrete(ACTION_SIZE)
        # 1D array, each element is a real value ranging from [0,10]
        self.observation_space = Box(low=0, high=10, shape=(FEATURE_SIZE,))
        self.world = MontezumaWorld()
        self.lifelong_step_cnt = 0
        self.reset()

    def reset(self):
        # how many times does the agent die
        self.lifelong_step_cnt += 1
        self.scenario = self.world.sample_scenario()
        self.state = self.scenario.init()
        self._game_status = 0
        self.stack_feature = np.stack(self.history * [self.state.features()], axis=1)
        if self.display:
            print('------------- New Game -----------------')
            self.state.visualize()
        return self.state.features()

    def step(self, action):
        self.lifelong_step_cnt += 1
        self._game_status, self._action_success, reward, self.state = self.state.step(action)
        if self.noLifeReward:
            if reward < 0:
                reward = 0
        x = self.state.features()
        self.stack_feature = np.append(self.stack_feature[:,1:], np.expand_dims(x, axis=1), axis = 1)
        return self.stack_feature, reward, (self._game_status!=0), None

    def render(self):
        print()
        self.state.visualize()

    def close(self):
        return None
