import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import copy
from queue import Queue, PriorityQueue
import heapq

import cv2
import argparse

import os
import time
import datetime
import pickle
import sys

from tm_wrapper import MontezumaWrapper_gym
import toy_Montezuma

from sys import getsizeof, stderr
try:
	from reprlib import repr
except ImportError:
	pass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
		
from skimage.transform import resize
from skimage.color import rgb2gray	
from skimage import io as skio
from skimage.transform import resize, downscale_local_mean
from skimage import img_as_float, img_as_ubyte, img_as_int

import pickle

#import scipy
import imageio
import ffmpy
import subprocess

import argparse

import warnings

from config import *


class basic_node(object):
	def __init__(self, total_reward=0):
		self.total_reward = total_reward

class visited_dict(object):
	def __init__(self):
		self.visited_nodes = dict()

	def is_visited(self, state_hash):
		return state_hash in self.visited_nodes	

	def add_node(self, state_hash, basic_node):
		self.visited_nodes[state_hash] = basic_node

	def get_total_reward(self, state_hash):
		return self.visited_nodes[state_hash].total_reward

	def set_total_reward(self, state_hash, new_reward):
		self.visited_nodes[state_hash].total_reward = new_reward

	def get_node(self, state_hash):
		return self.visited_nodes[state_hash]

	def len(self):
		return len(self.visited_nodes)		
	

class queue_node_data(object):
	def __init__(self, action_seq=[], basic_node=None, state_snapshot=None):
		self.action_seq = action_seq
		self.basic_node = basic_node
		self.state_snapshot = state_snapshot
		self.timestamp = -1000.0

	def add_reward(self, r):
		self.basic_node.total_reward += r

	def set_reward(self, new_reward):
		self.basic_node.total_reward = new_reward		

	def set_state_snapshot(self, state_snapshot):
		self.state_snapshot = state_snapshot

	def add_action(self, a):
		self.action_seq.append(a)	

	def set_time(self):
		self.timestamp = time.time()

	def set_action_seq(self, act_seq):
		self.action_seq = act_seq			


class queue_dict(object):
	def __init__(self):
		self.queue_data = dict()

	def is_in_queue(self, state_hash):
		return state_hash in self.queue_data	

	def add_node(self, state_hash, data_node):
		self.queue_data[state_hash] = data_node
	
	def del_node(self, state_hash):
		if state_hash in self.queue_data:
			return self.queue_data.pop(state_hash, None)
		else:
			print("Error: requested state is not in queue")
			return None

	def get_node(self, state_hash):
		return self.queue_data[state_hash]

	def len(self):
		return len(self.queue_data)

	def get_action_seq_len(self, state_hash):
		return len(self.queue_data[state_hash].action_seq)	

	def set_state_snapshot(self, state_hash, state_snapshot):
		self.queue_data[state_hash].set_state_snapshot(state_snapshot)

	def add_action(self, state_hash, a):
		self.queue_data[state_hash].add_action(a)	

	def set_time(self, state_hash):
		self.queue_data[state_hash].set_time()

	def set_action_seq(self, state_hash, act_seq):
		self.queue_data[state_hash].set_action_seq(act_seq)		

# large reward state gets high priority
class queue_node_priority(object):
	def __init__(self, state_hash, data):
		# hash code
		self.state_hash = state_hash
		self.data = data
	

	def __lt__(self, other):
		if strategy == "timestamp":
			return self.data.timestamp < other.data.timestamp
		else:
			if self.data.basic_node.total_reward < other.data.basic_node.total_reward:
				return True
			elif self.data.basic_node.total_reward == other.data.basic_node.total_reward:
				return self.data.timestamp < other.data.timestamp
			else:
				return False				

class priority_queue(object):
	def __init__(self):
		self.q = []

	def pop(self):
		return heapq.heappop(self.q)	

	def push(self, queue_node):	
		heapq.heappush(self.q, queue_node)

	def is_empty(self):
		return not self.q

	def len(self):
		return len(self.q)	

# downsample: 210*160 -> 11*8
# (height, width)
def get_representation(gray_img, return_vec=True, height_scale=20, width_scale=20):
	img = img_as_float(gray_img)
	img = downscale_local_mean(img, (height_scale, width_scale))

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		img = img_as_ubyte(img)

	if return_vec:
		img_vec = img.flatten()
		return img_vec	
	else:
		return img	
	
def get_orig_gray(gray_img):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		img = img_as_ubyte(gray_img)

	img_vec = img.flatten()

	return img_vec	

def get_state_hashcode(gray_obs, use_raw_image, downsample_scale=20):
	gray_obs = np.squeeze(gray_obs)
	if use_raw_image:
		state_hash = hash(get_orig_gray(gray_obs).tostring())
	else:	
		state_hash = hash(get_representation(gray_obs, return_vec=True, height_scale=downsample_scale, width_scale=downsample_scale).tostring())

	return state_hash

def save_obs_rep(observation, path, name):
	rep = get_representation(observation, return_vec=False, height_scale=10, width_scale=10)
	
	skio.imsave(os.path.join(path, name+'_obs.jpg'), observation)

	skio.imsave(os.path.join(path, name+'_rep.jpg'), rep)

	
def test_state_representation(env_name='MontezumaRevengeDeterministic-v4'):
	test_res_dir = os.path.join(exp_path, 'test_representation2')

	if not os.path.exists(test_res_dir):
		os.makedirs(test_res_dir)
	else:
		clear_dir(test_res_dir)	

	#epi_n = 2
	epi_n = 1
	act_n = 100

	env = gym.make(env_name)

	for i_episode in list(range(1,epi_n+1)):
		_ = env.reset()
		observation = env.unwrapped.ale.getScreenGrayscale()
		observation = np.squeeze(observation)

		save_obs_rep(observation, test_res_dir, '%d_0'%(i_episode))

		for t in list(range(1,act_n+1)):
			#env.render()
			action = env.action_space.sample()
			_, _, done, _ = env.step(action)
			observation = env.unwrapped.ale.getScreenGrayscale()
			observation = np.squeeze(observation)

			save_obs_rep(observation, test_res_dir, '%d_%d'%(i_episode, t))

			if done:
				print("Episode %d finished after %d timesteps"%(i_episode, t))
				break

def test_state_hashcode():
	test_res_dir = os.path.join(exp_path, 'test_state_hashcode')

	if not os.path.exists(test_res_dir):
		os.makedirs(test_res_dir)
	else:
		clear_dir(test_res_dir)	

	use_raw_image = True #True
	env = gym.make('MontezumaRevengeDeterministic-v4')

	_ = env.reset()
	init_state = env.unwrapped.ale.getScreenGrayscale()
	init_state = np.squeeze(init_state)

	init_state_hash = get_state_hashcode(init_state, use_raw_image)

	print(init_state_hash)


# for saving checkpoint: save references
class all_in_one(object):
	def __init__(self, my_priority_queue, my_queue_dict, my_visited_dict, state_cnt, step_cnt, best_score, best_action_seq, time_elapsed):
		self.my_priority_queue = my_priority_queue
		self.my_queue_dict = my_queue_dict
		self.my_visited_dict = my_visited_dict
		self.state_cnt = state_cnt
		self.step_cnt = step_cnt
		self.best_score = best_score
		self.best_action_seq = best_action_seq
		self.time_elapsed = time_elapsed

# for saving exploration results (save when a new highest score has been reached)
class explore_results(object):
	def __init__(self, best_score, best_action_seq):
		self.best_score = best_score
		self.best_action_seq = best_action_seq

# save checkpoint
def save_checkpoint(checkpoint_dir, my_priority_queue, my_queue_dict, my_visited_dict, state_cnt, step_cnt, best_score, best_action_seq, time_elapsed):
	saved_object = all_in_one(my_priority_queue, my_queue_dict, my_visited_dict, state_cnt, step_cnt, best_score, best_action_seq, time_elapsed)
	# protocol=-1 means highest protocol
	with open(os.path.join(checkpoint_dir, checkpoint_name), 'wb') as handle:
		pickle.dump(saved_object, handle, protocol=-1)


# save exploration results
def save_exploration_results(exp_results_dir, best_score, best_action_seq, name, timestamp=None):
	saved_object = explore_results(best_score, best_action_seq)
	if not timestamp:
		filename = '%s.pkl'%(name)
	else:
		filename = '%s_%s.pkl'%(name, timestamp)

	with open(os.path.join(exp_results_dir, filename), 'wb') as handle:
		pickle.dump(saved_object, handle, protocol=-1)	

	print('Best trajectory saved')


# replay best trajectory after exploration
def replay_best_trajectory(env_name, traj_dir, traj_name, timestamp=None):
	if not timestamp:
		file_name = '%s.pkl'%(traj_name)
	else:	
		file_name = '%s_%s.pkl'%(traj_name, timestamp)

	# load best trajectory
	with open(os.path.join(traj_dir, file_name), 'rb') as handle:
		saved_object = pickle.load(handle)

	best_score = saved_object.best_score
	best_action_seq = saved_object.best_action_seq

	if not best_action_seq:
		print('Best results are empty')
		return

	# create environment
	env = gym.make(env_name)
	env.reset()
	# replay actions and save observations as .gif
	with imageio.get_writer(os.path.join(traj_dir, '%s_%s.gif'%(traj_name, timestamp)), mode='I') as writer:
		for action in best_action_seq:
			observation, reward, done, _ = env.step(action)
			writer.append_data(observation)

	cmd = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'panic', '-y', '-i', os.path.join(traj_dir, '%s_%s.gif'%(traj_name, timestamp)), os.path.join(traj_dir, '%s_%s.avi'%(traj_name, timestamp))]
	retcode = subprocess.call(cmd)
	if not retcode == 0:
		raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd))) 

	print('Replayed animation saved')
	

# replay best trajectory after exploration
def test_replay_best_trajectory(env_name, traj_dir, traj_name, timestamp=None):
	if not timestamp:
		file_name = '%s.pkl'%(traj_name)
	else:	
		file_name = '%s_%s.pkl'%(traj_name, timestamp)

	# load best trajectory
	with open(os.path.join(traj_dir, file_name), 'rb') as handle:
		saved_object = pickle.load(handle)

	best_score = saved_object.best_score
	best_action_seq = saved_object.best_action_seq

	if not best_action_seq:
		print('Best results are empty')
		return

	# create environment
	env = gym.make(env_name)
	env.reset()

	total_reward = 0
	# replay actions and save observations as .gif
	for action in best_action_seq:
		observation, reward, done, _ = env.step(action)
		total_reward += reward

	print("Best score: %d"%(best_score))
	print("Total score: %d"%(total_reward))	


def is_gameover(one_life, game_done, current_lives, last_lives):
	if not one_life:
		return game_done
	else:
		if current_lives < last_lives:
			return True
		else:
			return False	

# my implementation of bellman-ford
def my_bellman_ford(env_name='MontezumaRevengeDeterministic-v4', load_checkpoint_name="", use_raw_image=False, replay_action=False, reduce_action=False, random_order=False, downsample_scale=20, one_life=False, score_threshold=2000000, log_state_interval=1000, save_state_interval=10000):
	# paths
	log_dir = os.path.join(bellman_ford_dir, 'logs')		
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	exp_results_dir = os.path.join(bellman_ford_dir, 'explore_results')
	if not os.path.exists(exp_results_dir):
		os.makedirs(exp_results_dir)
	checkpoint_dir = os.path.join(bellman_ford_dir, 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)	
	

	### redirect standard output	
	### name log file as current time
	current_time = time.time()
	log_st = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
	logfile = open(os.path.join(log_dir,'bellman_ford_'+log_st+'.txt'), 'w')
	sys.stdout = Logger(sys.stdout, logfile)

	# create environment
	if env_name == "SimpleMontezuma":
		env = MontezumaWrapper_gym(history=4)
	else:	
		env = gym.make(env_name).env

	print('------------------------------------------')
	print("Searching algorithm: Bellman-ford")
	print("Environment name: "+env_name)

	# start timer	
	start_time = time.time()
	start_st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')
	print('Start time: '+start_st)
	print('------------------------------------------')

	# create exp dir for this time experiment
	cur_exp_dir = os.path.join(exp_results_dir, start_st)
	if not os.path.exists(cur_exp_dir):
		os.makedirs(cur_exp_dir)

	# load checkpoint
	if load_checkpoint_name:
		with open(os.path.join(checkpoint_dir, load_checkpoint_name), 'rb') as handle:
			saved_object = pickle.load(handle)

		my_priority_queue = saved_object.my_priority_queue
		my_queue_dict = saved_object.my_queue_dict
		my_visited_dict = saved_object.my_visited_dict
		state_cnt = saved_object.state_cnt
		step_cnt = saved_object.step_cnt
		best_score = saved_object.best_score
		best_action_seq = saved_object.best_action_seq
		prev_time_elapsed = saved_object.time_elapsed

		print('Checkpoint loaded: %s'%(load_checkpoint_name))
		print('State explored: %d'%(state_cnt))
		print('Emulator steps: %d'%(step_cnt))
		print('Best instant score so far: %.1f'%(best_score))
		print('------------------------------------------')

		# reset enviroment [otherwise, we cannot step from restored environment]
		env.reset()
	# start from the very beginning	
	else:
		# create data structures
		my_queue_dict = queue_dict()
		my_priority_queue = priority_queue()
		my_visited_dict = visited_dict()
		# state explored
		state_cnt = 0
		# emulator steps
		step_cnt = 0
		# best score achieved so far
		best_score = 0
		# best action sequence so far
		best_action_seq = []
		# time elasped before the loaded checkpoint
		prev_time_elapsed = 0
		# get hashcode for initial state
		_ = env.reset()
		init_gray_obs = env.unwrapped.ale.getScreenGrayscale()
		init_state_hash = get_state_hashcode(init_gray_obs, use_raw_image, downsample_scale)

		# initialize data structures
		n0_basic = basic_node()
		my_visited_dict.add_node(init_state_hash, n0_basic)
		if replay_action:
			n0_data = queue_node_data(action_seq=[], basic_node=n0_basic, state_snapshot=None)
		else:	
			n0_data = queue_node_data(action_seq=[], basic_node=n0_basic, state_snapshot=copy.deepcopy(env.unwrapped.clone_full_state()))
		
		my_queue_dict.add_node(init_state_hash, n0_data)
		n0_data.set_time()
		n0 = queue_node_priority(init_state_hash, n0_data)
		my_priority_queue.push(n0)
		print('Start from beginning')
		print('------------------------------------------')

	# get possible actions
	action_size = env.action_space.n

	# verbose basic setting
	print('Basic settings:')
	if reduce_action:
		print('Action space reduced: 8')
	else:
		print('Action space: %d'%(action_size))

	print('Score threshold: %.1f'%(score_threshold))
	
	if use_raw_image:
		if env_name == "SimpleMontezuma":
			print('State space: 8*11 raw image')
		else:	
			print('State space: 210*160 raw image')
	else:
		down_height = int(math.ceil(210 / downsample_scale))
		down_width = int(math.ceil(160 / downsample_scale)) 
		print('State space: %d*%d downsampled image'%(down_height, down_width))
	if replay_action:
		print('Environment restore method: replay action sequence')
	else:
		print('Environment restore method: restore state')
	if random_order:
		print('Action is in random order')
	else:
		print('Action is in fixed order')
	if one_life:
		print('Game over after losing 1 life')
	else:
		print('Game over after losing all 6 lives')	

	print('Searching strategy: %s'%(strategy))		

	print('------------------------------------------')

	interval_start_time = time.time()
	last_step_cnt = step_cnt
	
	# start searching
	while not my_priority_queue.is_empty() and best_score < score_threshold:
		# pop max reward node as u and explore from it
		u = my_priority_queue.pop()
		u_data = my_queue_dict.del_node(u.state_hash)
		
		# Breadth first search from u
		if reduce_action:
			all_actions = [0,1,2,3,4,5,11,12]
		else:	
			all_actions = np.arange(action_size)
		
		if random_order:
			np.random.shuffle(all_actions)
 
 		# try all actions
		for action in all_actions:
			# restore env to u
			# replay actions
			if replay_action:
				# reset (reset is not counted for one step)
				_ = env.reset()
				for act in u_data.action_seq:
					_, _, _, _ = env.step(act)
					step_cnt += 1
			# load env state from RAM	
			# restore is not counted for one step	
			else:
				env.unwrapped.restore_full_state(copy.deepcopy(u_data.state_snapshot))

			last_lives = env.unwrapped.ale.lives()

			# one step forward
			_, reward, done, info = env.step(action)
			gray_obs = env.unwrapped.ale.getScreenGrayscale()
			step_cnt += 1

			# look at life changes
			current_lives = env.unwrapped.ale.lives()
			
			# get current reward
			new_total_reward = u_data.basic_node.total_reward + reward

			# update highest instant reward
			if new_total_reward > best_score:
				print("Highest instant score achieved: %.1f at step %d"%(new_total_reward, step_cnt))
				best_score = new_total_reward
				best_action_seq = copy.deepcopy(u_data.action_seq)
				best_action_seq.append(action)

				print("Previous score: %d"%(u_data.basic_node.total_reward))
				print("Action %d gets score: %d"%(action, reward))
				# save final exploration results
				time_score_st = str(best_score)+"_"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
				save_exploration_results(cur_exp_dir, best_score, best_action_seq, "best", time_score_st)
				if env_name != "SimpleMontezuma":
					# replay best instant trajectory
					replay_best_trajectory(env_name=env_name, traj_dir=cur_exp_dir, traj_name="best", timestamp=time_score_st)
				
				print('------------------------------------------')	

			# game is continued, continue exploration	
			if not is_gameover(one_life, done, current_lives, last_lives):
				# hash current state
				state_hash = get_state_hashcode(gray_obs, use_raw_image, downsample_scale)

				# if current state has not been visited
				if not my_visited_dict.is_visited(state_hash):
					# create queue data
					n_basic = basic_node(new_total_reward)
					if replay_action:
						n_data = queue_node_data(action_seq=copy.deepcopy(u_data.action_seq), basic_node=n_basic, state_snapshot=None)
					else:
						n_data = queue_node_data(action_seq=copy.deepcopy(u_data.action_seq), basic_node=n_basic, state_snapshot=copy.deepcopy(env.unwrapped.clone_full_state()))
					
					n_data.add_action(action)
					# record as visited
					my_visited_dict.add_node(state_hash, n_basic)
					# push to queue
					my_queue_dict.add_node(state_hash, n_data)
					n_data.set_time()
					n_node = queue_node_priority(state_hash, n_data)
					my_priority_queue.push(n_node)
				# if current state has been visited and get a higher reward
				elif new_total_reward > my_visited_dict.get_total_reward(state_hash):
					# update reward
					my_visited_dict.set_total_reward(state_hash, new_total_reward)

					# if not in queue, create queue data and push to queue
					if not my_queue_dict.is_in_queue(state_hash):
						# create queue data
						if replay_action:
							n_data = queue_node_data(action_seq=copy.deepcopy(u_data.action_seq), basic_node=my_visited_dict.get_node(state_hash), state_snapshot=None)
						else:
							n_data = queue_node_data(action_seq=copy.deepcopy(u_data.action_seq), basic_node=my_visited_dict.get_node(state_hash), state_snapshot=copy.deepcopy(env.unwrapped.clone_full_state()))
						
						n_data.add_action(action)
						n_data.set_time()

						# push
						my_queue_dict.add_node(state_hash, n_data)
						n_node = queue_node_priority(state_hash, n_data)
						my_priority_queue.push(n_node)
					# is in queue, update queue data
					else:
						if not replay_action:
							my_queue_dict.set_state_snapshot(state_hash=state_hash, state_snapshot=copy.deepcopy(env.unwrapped.clone_full_state()))
							
						
						my_queue_dict.set_action_seq(state_hash=state_hash, act_seq=copy.deepcopy(u_data.action_seq))
						my_queue_dict.add_action(state_hash, action)
						my_queue_dict.set_time(state_hash)
		
		# completed exploration of u
		state_cnt += 1

		# verbose 
		if state_cnt % log_state_interval == 0:
			print('Emulator steps: %d'%(step_cnt))
			print('Searched states: %d'%(state_cnt))
			print('Visited states: %d'%(my_visited_dict.len()))
			print('Queue length: %d [%d]'%(my_priority_queue.len(), my_queue_dict.len()))
			print('Best instant score so far: %.1f'%(best_score))
			seconds_elapsed = time.time()-interval_start_time
			hour_elapsed = seconds_elapsed / 3600
			print('Time elapsed: %s'%(str(datetime.timedelta(seconds=seconds_elapsed))))
			print('States per hour: %d'%(log_state_interval / hour_elapsed))
			print('Steps per hour: %d'%((step_cnt - last_step_cnt) / hour_elapsed))
			print('------------------------------------------')
			# next turn
			interval_start_time = time.time()
			last_step_cnt = step_cnt

		# save checkpoint
		if state_cnt % save_state_interval == 0:
			time_elapsed = time.time() - start_time			
			save_checkpoint(checkpoint_dir, my_priority_queue, my_queue_dict, my_visited_dict, state_cnt, step_cnt, best_score, best_action_seq, time_elapsed)
			print('Searched states: %d, checkpoint saved'%(state_cnt))
			print('------------------------------------------')		
		
	end_time = time.time()
	end_st = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')	
	# Final report		
	print('Summary:')
	print('Best instant score: %.1f'%(best_score))
	print('Emulator steps: %d'%(step_cnt))
	print('Searched states: %d'%(state_cnt))
	print('Visited states: %d'%(my_visited_dict.len()))
	print('Queue length: %d [%d]'%(my_priority_queue.len(), my_queue_dict.len()))
	print('Start time: '+start_st)
	print('End time: '+end_st)
	total_seconds_elapsed = prev_time_elapsed + end_time - start_time
	total_hour_elapsed = total_seconds_elapsed / 3600
	print('Total exploration time: '+str(datetime.timedelta(seconds=total_seconds_elapsed)))
	print('States per hour: %d'%(state_cnt / total_hour_elapsed))
	print('Steps per hour: %d'%(step_cnt / total_hour_elapsed))
	print('------------------------------------------')
	
	# rename checkpoint
	if os.path.exists(os.path.join(checkpoint_dir, checkpoint_name)):
		os.rename(os.path.join(checkpoint_dir, checkpoint_name), os.path.join(checkpoint_dir, 'all_%s.pkl'%(log_st)))


	env.close()
	print('Done.')	


if __name__ == "__main__":
	'''
	'''
	parser = argparse.ArgumentParser(description='Searching in state space')
	parser.add_argument('--score', default=2000000, type=float, help='highest scores: 2000000 go explore best, 400000 go explore on average, 35000 human expert, 5000 human average, 400 1st room')
	parser.add_argument('--env', default='MontezumaRevengeDeterministic-v4', type=str, help='environment name: SimpleMontezuma | MontezumaRevengeDeterministic-v4')
	parser.add_argument('--loginterval', default=1000, type=int, help='state log interval')
	parser.add_argument('--saveinterval', default=10000, type=int, help='state save interval: 5000, 10000')
	parser.add_argument('--checkpoint', default="", type=str, help='name of checkpoint')
	parser.add_argument('--rawimage', action='store_true', help='state space is raw image rather than downsampled image')
	parser.add_argument('--replay', action='store_true', help='replay action rather than restore state')
	parser.add_argument('--actionreduce', action='store_true', help='reduce action space to feasible ones')
	parser.add_argument('--randomact', action='store_true', help='perform actions in random order in each step')
	parser.add_argument('--downscale', default=20, type=int, help='downsample scale')
	parser.add_argument('--onelife', action='store_true', help='game over when life is lost once')
	parser.add_argument('--strategy', default="timestamp", type=str, help='timestamp | score')

	args = parser.parse_args()
	
	# global paramters
	strategy = args.strategy
	
	my_bellman_ford(env_name=args.env, load_checkpoint_name=args.checkpoint, use_raw_image=args.rawimage, replay_action=args.replay, reduce_action=args.actionreduce, random_order=args.randomact, downsample_scale=args.downscale, one_life=args.onelife, score_threshold=args.score, log_state_interval=args.loginterval, save_state_interval=args.saveinterval)
	
	#test_state_representation()
	#test_state_hashcode()
	#cur_exp_dir = os.path.join(os.path.join(bellman_ford_dir, 'explore_results'), "2019-02-07_18:58:24")
	#test_replay_best_trajectory(env_name="MontezumaRevengeDeterministic-v4", traj_dir=cur_exp_dir, traj_name="best", timestamp="700.0_2019-02-07_20:02:35")
	
	
	