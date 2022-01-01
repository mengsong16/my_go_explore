# downsample: 210*160 -> 11*8
# (height, width)
def get_representation(gray_img, return_vec=True, quantize_pixel=True, height_scale=20, width_scale=20):
	img = img_as_float(gray_img)
	img = downscale_local_mean(img, (height_scale, width_scale))

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		img = img_as_ubyte(img)
	
	if quantize_pixel:
		with warnings.catch_warnings():	
			warnings.simplefilter("ignore")
			img = img.astype(float)
			img = img / 32 
			img = np.rint(img).astype(int)

	if return_vec:
		img_vec = img.flatten()
		return img_vec	
	else:
		return img

def save_obs_rep(observation, path, name):
	rep = get_representation(observation, return_vec=False)
	
	skio.imsave(os.path.join(path, name+'_obs.jpg'), observation)

	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		rep = img_as_int(rep*32)
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