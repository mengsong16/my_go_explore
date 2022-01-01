import os
import shutil

# paths
project_path = "/home/meng/my_go_explore"
exp_path = os.path.join(project_path, 'exp')
if not os.path.exists(exp_path):
	os.makedirs(exp_path)
go_explore_dir = os.path.join(exp_path, 'go_explore')
if not os.path.exists(go_explore_dir):
	os.makedirs(go_explore_dir)
bellman_ford_dir = os.path.join(exp_path, 'bellman_ford')
if not os.path.exists(bellman_ford_dir):
	os.makedirs(bellman_ford_dir)

checkpoint_name = 'all.pkl'


# functions
class Logger(object):
	def __init__(self, f1, f2):
		self.f1, self.f2 = f1, f2

	def write(self, msg):
		self.f1.write(msg)
		self.f2.write(msg)
		
	def flush(self):
		pass

# clear directory
def clear_dir(folder):
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print(e)
	return				

