import gym

while True:
	env = gym.make('FetchPickAndPlace-v0')
	env.reset()
	observation, reward, done, info = env.step(0)
	print (observation)
	
	env.render()

