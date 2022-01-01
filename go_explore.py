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
from skimage import img_as_float, img_as_ubyte

import pickle

#import scipy
import imageio
import ffmpy
import subprocess

import argparse

import warnings

from config import *
from random import choices

class archive_node(object):
	def __init__(self, total_reward=0, action_seq=[], state_snapshot=None):
		self.n_chosen = 0
		self.n_seen = 0
		self.n_chosen_since_new = 0
		self.cell_score = 0

		self.total_reward = total_reward
		self.action_seq = action_seq
		self.state_snapshot = state_snapshot

	def add_reward(self, r):
		self.total_reward += r

	def set_reward(self, new_reward):
		self.total_reward = new_reward		

	def set_state_snapshot(self, state_snapshot):
		self.state_snapshot = state_snapshot

	def add_action(self, a):
		self.action_seq.append(a)	

	def set_action_seq(self, act_seq):
		self.action_seq = act_seq

	def get_action_seq_len(self):
		return len(self.action_seq)	

	def compute_count_score(self, v, weight):
		if weight == 0:
			return epsilon2
		else:
			return weight * math.pow(1.0/(v+epsilon1), pa) + epsilon2	

	def compute_cell_score(self):
		count_score_sum = self.compute_count_score(self.n_chosen, weight_chosen) + self.compute_count_score(self.n_seen, weight_seen) + self.compute_count_score(self.n_chosen_since_new, weight_chosen_since_new)
		self.cell_score = count_score_sum + 1.0		


class archive(object):
	def __init__(self):
		self.queue_data = dict()	

	def add_node(self, state_hash, data_node):
		self.queue_data[state_hash] = data_node
	
	def choose_node(self):
		keys = list(self.queue_data.keys())
		weights = [1] * self.len()

		# use cell score if not uniform
		if strategy != "uniform":
			i = 0
			for node in list(self.queue_data.values()):
				weights[i] = node.cell_score
				i = i + 1

		choose_key = choices(keys, weights)[0]

		return self.queue_data[choose_key]
		
	def get_node(self, state_hash):
		return self.queue_data[state_hash]

	def len(self):
		return len(self.queue_data)

	def get_action_seq_len(self, state_hash):
		return len(self.queue_data[state_hash].action_seq)	

	def add_action(self, state_hash, a):
		self.queue_data[state_hash].add_action(a)	

	def set_action_seq(self, state_hash, act_seq):
		self.queue_data[state_hash].set_action_seq(act_seq)	

	def set_state_snapshot(self, state_hash, state_snapshot):
		self.queue_data[state_hash].set_state_snapshot(state_snapshot)

	def is_empty(self):
		return not self.queue_data

	def is_visited(self, state_hash):
		return state_hash in self.queue_data	

	def get_total_reward(self, state_hash):
		return self.queue_data[state_hash].total_reward

	def set_total_reward(self, state_hash, new_reward):
		self.queue_data[state_hash].total_reward = new_reward	

	def add_n_seen(self, state_hash):
		self.queue_data[state_hash].n_seen += 1

	def add_n_chosen(self, state_hash):
		self.queue_data[state_hash].n_chosen += 1	

	def add_n_chosen_since_new(self, state_hash):
		self.queue_data[state_hash].n_chosen_since_new += 1

	def reset_n_chosen(self, state_hash):
		self.queue_data[state_hash].n_chosen = 0	

	def reset_n_chosen_since_new(self, state_hash):
		self.queue_data[state_hash].n_chosen_since_new = 0	

	def compute_cell_score(self, state_hash):
		self.queue_data[state_hash].compute_cell_score()					
	

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
	def __init__(self, my_archive, state_cnt, step_cnt, best_score, best_action_seq, time_elapsed):
		self.my_archive = my_archive
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
def save_checkpoint(checkpoint_dir, my_archive, state_cnt, step_cnt, best_score, best_action_seq, time_elapsed):
	saved_object = all_in_one(my_archive, state_cnt, step_cnt, best_score, best_action_seq, time_elapsed)
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
	

def is_gameover(one_life, game_done, current_lives, last_lives):
	if not one_life:
		return game_done
	else:
		if current_lives < last_lives:
			return True
		else:
			return False

# explore stage in go-explore
def go_explore(env_name='MontezumaRevengeDeterministic-v4', load_checkpoint_name="", use_raw_image=False, replay_action=False, reduce_action=False, downsample_scale=20, one_life=False, explore_trajectory_len=100, score_threshold=2000000, log_state_interval=1000, save_state_interval=10000):
	# paths	
	log_dir = os.path.join(go_explore_dir, 'logs')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	exp_results_dir = os.path.join(go_explore_dir, 'explore_results')
	if not os.path.exists(exp_results_dir):
		os.makedirs(exp_results_dir)
	checkpoint_dir = os.path.join(go_explore_dir, 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)	

	### redirect standard output	
	### name log file as current time
	current_time = time.time()
	log_st = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
	logfile = open(os.path.join(log_dir,'go_explore_'+log_st+'.txt'), 'w')
	sys.stdout = Logger(sys.stdout, logfile)

	# create environment
	if env_name == "SimpleMontezuma":
		env = MontezumaWrapper_gym(history=4)
	else:	
		env = gym.make(env_name).env

	print('------------------------------------------')
	print("Searching algorithm: go-explore")
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

		my_archive = saved_object.my_archive
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
		my_archive = archive()
		# state explored
		state_cnt = 0
		# emulator steps
		step_cnt = 0
		# best score achieved
		best_score = 0
		# best action sequence
		best_action_seq = []
		# time elasped before the loaded checkpoint
		prev_time_elapsed = 0
		# get hashcode for initial state
		_ = env.reset()
		init_gray_obs = env.unwrapped.ale.getScreenGrayscale()
		init_state_hash = get_state_hashcode(init_gray_obs, use_raw_image, downsample_scale)

		# initialize data structures
		n0 = archive_node()
	
		# update score for node 0
		if strategy != "uniform":
			n0.n_seen += 1
			n0.compute_cell_score()

		if not replay_action:
			n0.set_state_snapshot(copy.deepcopy(env.unwrapped.clone_full_state()))

		my_archive.add_node(init_state_hash, n0)
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

	print('Exploration trajectory length: %d'%(explore_trajectory_len))


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
	if one_life:
		print('Game over after losing 1 life')
	else:
		print('Game over after losing all 6 lives')	

	
	print('Searching strategy: %s'%(strategy))	
	
						
	print('------------------------------------------')

	interval_start_time = time.time()
	last_step_cnt = step_cnt
	# start searching
	while best_score < score_threshold:
		# choose node u and explore from it
		u_data = my_archive.choose_node()
		if strategy != "uniform":
			u_data.n_chosen += 1
			u_data.n_chosen_since_new += 1
			#u_data.compute_cell_score()
		# restore env to u
		# replay actions
		if replay_action:
			# reset
			_ = env.reset()
			for act in u_data.action_seq:
				_, _, _, _ = env.step(act)
				step_cnt += 1
		# load state from RAM		
		else:
			env.unwrapped.restore_full_state(copy.deepcopy(u_data.state_snapshot))		

		# Starting from u, randomly explore some steps
		# record action and accumulated reward along the trajectory
		new_total_reward = u_data.total_reward
		traj_action_seq = copy.deepcopy(u_data.action_seq)
		
		for t in list(range(explore_trajectory_len)):
			last_lives = env.unwrapped.ale.lives()

			# random one step forward
			if not reduce_action:
				action = env.action_space.sample()
			else:
				action = random.choice([0,1,2,3,4,5,11,12])	

			_, reward, done, info = env.step(action)
			gray_obs = env.unwrapped.ale.getScreenGrayscale()
			step_cnt += 1

			# look at life changes
			current_lives = env.unwrapped.ale.lives()

			# update new reward and new action sequence, sequence length along the trajectory
			new_total_reward += reward
			traj_action_seq.append(action)

			# update highest instant reward
			if new_total_reward > best_score:
				print("Highest instant score achieved: %.1f at step %d"%(new_total_reward, step_cnt))
				best_score = new_total_reward
				best_action_seq = copy.deepcopy(traj_action_seq)
				# save final exploration results
				time_score_st = str(best_score)+"_"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
				save_exploration_results(cur_exp_dir, best_score, best_action_seq, "best", time_score_st)
				if env_name != "SimpleMontezuma":
					# replay best instant trajectory
					replay_best_trajectory(env_name=env_name, traj_dir=cur_exp_dir, traj_name="best", timestamp=time_score_st)
				
				print('------------------------------------------')	
			
			# game over (lost all lives, arrived terminate state)
			if is_gameover(one_life, done, current_lives, last_lives):
				# stop this trajectory
				break		
			# continue exploration trajectory		
			else:
				# hash current state
				state_hash = get_state_hashcode(gray_obs, use_raw_image, downsample_scale)

				# if current state has not been visited
				if not my_archive.is_visited(state_hash):
					# create new node
					if replay_action:
						n_node = archive_node(total_reward=new_total_reward, action_seq=copy.deepcopy(traj_action_seq), state_snapshot=None)
					else:
						n_node = archive_node(total_reward=new_total_reward, action_seq=copy.deepcopy(traj_action_seq), state_snapshot=env.unwrapped.clone_full_state())
		
					# put on archive
					my_archive.add_node(state_hash, n_node)

					# u leads to new cell, update u score
					if strategy != "uniform":
						u_data.n_chosen_since_new = 0
					
				# if current state has been on archive and get a higher reward or shorter trajectory
				elif (new_total_reward > my_archive.get_total_reward(state_hash)) or ((new_total_reward == my_archive.get_total_reward(state_hash)) and (len(traj_action_seq) < my_archive.get_action_seq_len(state_hash))):
					# update score, action sequence and snapshot
					my_archive.set_total_reward(state_hash, new_total_reward)
					if not replay_action:
						my_archive.set_state_snapshot(state_hash=state_hash, state_snapshot=copy.deepcopy(env.unwrapped.clone_full_state()))
					
					my_archive.set_action_seq(state_hash=state_hash, act_seq=copy.deepcopy(traj_action_seq))
					
					if strategy != "uniform":
						# reset scores of current cell to 0 since current cell is updated
						my_archive.reset_n_chosen(state_hash)
						my_archive.reset_n_chosen_since_new(state_hash)
						# u leads to better cell, update u score
						u_data.n_chosen_since_new = 0
						

				# current cell seen + 1
				if strategy != "uniform":
					my_archive.add_n_seen(state_hash)
					my_archive.compute_cell_score(state_hash)
							

		# completed exploration of u
		state_cnt += 1
		u_data.compute_cell_score()

		# verbose 
		if state_cnt % log_state_interval == 0:
			print('Emulator steps: %d'%(step_cnt))
			print('Searched states: %d'%(state_cnt))
			print('Visited states: %d'%(my_archive.len()))
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
			save_checkpoint(checkpoint_dir, my_archive, state_cnt, step_cnt, best_score, best_action_seq, time_elapsed)
			print('Searched states: %d, checkpoint saved'%(state_cnt))
			print('------------------------------------------')		
		
	end_time = time.time()
	end_st = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')	
	# Final report		
	print('Summary:')
	print('Best instant score: %.1f'%(best_score))
	print('Emulator steps: %d'%(step_cnt))
	print('Searched states: %d'%(state_cnt))
	print('Visited states: %d'%(my_archive.len()))
	print('Start time: '+start_st)
	print('End time: '+end_st)
	total_seconds_elapsed = prev_time_elapsed + end_time-start_time
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
	
	parser = argparse.ArgumentParser(description='Searching in state space')
	parser.add_argument('--score', default=2000000, type=float, help='highest scores: 2000000 go explore best, 400000 go explore on average, 35000 human expert, 5000 human average, 400 1st room')
	parser.add_argument('--env', default='MontezumaRevengeDeterministic-v4', type=str, help='environment name: SimpleMontezuma | MontezumaRevengeDeterministic-v4')
	parser.add_argument('--depth', default=100, type=int, help='trajectory length from explored state')
	parser.add_argument('--loginterval', default=100, type=int, help='state log interval')
	parser.add_argument('--saveinterval', default=500, type=int, help='state save interval: 5000, 10000')
	parser.add_argument('--checkpoint', default="", type=str, help='name of checkpoint')
	parser.add_argument('--rawimage', action='store_true', help='state space is raw image rather than downsampled image')
	parser.add_argument('--replay', action='store_true', help='replay action rather than restore state')
	parser.add_argument('--actionreduce', action='store_true', help='reduce action space to feasible ones')
	parser.add_argument('--downscale', default=20, type=int, help='downsample scale')
	parser.add_argument('--onelife', action='store_true', help='game over when life is lost once')
	parser.add_argument('--strategy', default="goexplore", type=str, help='uniform | goexplore')


	args = parser.parse_args()

	# global paramters
	strategy = args.strategy
	weight_chosen = 0.1
	weight_seen = 0.3
	weight_chosen_since_new = 0
	pa = 0.5
	epsilon1 = 0.001
	epsilon2 = 0.00001

	
	go_explore(env_name=args.env, load_checkpoint_name=args.checkpoint, use_raw_image=args.rawimage, replay_action=args.replay, reduce_action=args.actionreduce, downsample_scale=args.downscale, one_life=args.onelife, explore_trajectory_len=args.depth, score_threshold=args.score, log_state_interval=args.loginterval, save_state_interval=args.saveinterval)
	
	
	
	