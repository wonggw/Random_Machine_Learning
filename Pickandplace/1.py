import gym
import random
import numpy as np
import tensorflow as tf
import model as M
from collections import deque 
import memory

def actor(inp,scope):
	with tf.variable_scope(scope):
		mod = M.Model(inp,[None,4])
		mod.fcLayer(32,activation=M.PARAM_LRELU)
		mod.fcLayer(64,activation=M.PARAM_LRELU	)
		mod.fcLayer(128,activation=M.PARAM_LRELU)
		mod.fcLayer(1,activation=M.PARAM_TANH)
	return mod.get_current_layer()

def crit(inp,scope,reuse=False):
	with tf.variable_scope(scope,reuse=reuse):
		mod = M.Model(inp,[None,5])
		mod.fcLayer(32,activation=M.PARAM_LRELU)
		mod.fcLayer(64,activation=M.PARAM_LRELU)
		mod.fcLayer(128,activation=M.PARAM_LRELU)
		mod.fcLayer(1)
	return mod.get_current_layer()

def build_graph():
	envholder = tf.placeholder(tf.float32,[None,4])
	envholder2 = tf.placeholder(tf.float32,[None,4])
	reward_holder = tf.placeholder(tf.float32,[None,1])
	actholder = tf.placeholder(tf.float32,[None,1])
	terminated_holder = tf.placeholder(tf.float32,[None,1])
	
	#policy gradient
	#S_0 for actor_1
	a_eval = actor(envholder,'a1')
	#S_1 for actor_1
	a_real = actor(envholder2,'a2')

	#combine actor with env
	env_act1 = tf.concat([envholder,a_eval],axis=-1) #S_0
	env_act2 = tf.concat([envholder2,a_real],axis=-1) #S_1
	env_act3 = tf.concat([envholder,actholder],axis=-1)

	#DQN
	c_eval = crit(env_act1,'c1') #S_0
	c_real = crit(env_act2,'c2') #S_1
	c_real2 = crit(env_act1,'c2',True)
	c_eval2 = crit(env_act3,'c1',True)

	var_a1 = M.get_trainable_vars('a1')
	var_a2 = M.get_trainable_vars('a2')
	var_c1 = M.get_trainable_vars('c1')
	var_c2 = M.get_trainable_vars('c2')

	q_target = EPS*c_real*terminated_holder + reward_holder
	c_loss = tf.reduce_mean(tf.square(c_eval2 - q_target))

	a_loss = -tf.reduce_mean(c_real2)

	train_c = tf.train.RMSPropOptimizer(0.002).minimize(c_loss,var_list=var_c1)
	train_a = tf.train.RMSPropOptimizer(0.001).minimize(a_loss,var_list=var_a1) #maximize c_real2 , env_act1

	assign_a = soft_assign(var_a2,var_a1,0.5)
	assign_c = soft_assign(var_c2,var_c1,0.5)

	assign_a0 = assign(var_a2,var_a1)
	assign_c0 = assign(var_c2,var_c1)

	return [envholder,envholder2,reward_holder,actholder,terminated_holder],a_eval,[c_loss,a_loss],[train_c,train_a],[assign_c,assign_a],[assign_c0,assign_a0],a_eval

def soft_assign(old,new,tau):
	assign_op = [tf.assign(i,tau*j+(1-tau)*i) for i,j in zip(old,new)]
	return assign_op

def assign(old,new):
	assign_op = [tf.assign(i,j) for i,j in zip(old,new)]
	return assign_op

# Constants
EXPLORE = 100
TRAIN =1000
BSIZE = 32
frame_count = 0
env0 =[]
rw = 0
act = 0	
act1 = 0
# Explore and training
episode = 0
GAMMA = 0.99
var = 5
EPS = 0.9


#training_data = initial_population()
holders, action, losses, train_steps, assign, init_assign,q_target = build_graph()


env = gym.make('CartPole-v1')
observation = env.reset()

#memory = memory()

with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	saver = tf.train.Saver()
	observation, reward, done, info = env.step(0)
	#print (observation)
	env0 = [observation[0],observation[1],observation[2],observation[3]] *1

	while True:
		act = sess.run(action,feed_dict={holders[0]:[env0]})
		act = act[0]
		act = np.random.normal(act,var)
		act = np.clip(act,-1.,1.)
		if act >0:
			act1 = 1
		else:
			act1 = 0
		#print (act)
		observation, reward, done, info = env.step(int(act1))
		env1 = env0[4:] + [observation[0],observation[1],observation[2],observation[3]]

		if done == True:
			terminated = 0
		else: 
			terminated = 1
		act1 = [act1]
		memory.push([env0,env1,act,[reward],[terminated]])

		if rw>100:
			memory.push_prior([env0,env1,act,[reward],[terminated]])

		if terminated==0:
			env.reset()
			observation, reward, done, info = env.step(0)
			env0 = [observation[0],observation[1],observation[2],observation[3]] *1
			var = var*0.99
			episode += 1
			rw = 0
		else:
			env0 = env1
			rw +=1

		frame_count += 1
	
		# training
		if frame_count>=10000:
			train_batch = memory.next_batch(BSIZE)
			s0_batch = [i[0] for i in train_batch]
			# print(s0_batch[0])
			s1_batch = [i[1] for i in train_batch]
			a_batch = [i[2] for i in train_batch]
			rw_batch = [i[3] for i in train_batch]
			t_batch = [i[4] for i in train_batch]
			feed_d = {holders[0]:s0_batch,holders[1]:s1_batch,holders[2]:rw_batch,holders[3]:a_batch,holders[4]:t_batch}
			c_loss, a_loss, _,_ = sess.run(losses+train_steps,feed_dict=feed_d)
			# c_loss, a_loss = sess.run(losses,feed_dict=feed_d)
			if frame_count%100==0:
				sess.run(assign)

			if frame_count>14000:
				env.render()
	
			if frame_count%100000==0:
				saver.save(sess,'./model/%d.ckpt'%(frame_count))
			if frame_count%100==0:
				print('Episode:%d\tFrame:%d\tC_Loss:%.4f\tA_Loss:%.4f\tQ:%d\tEpsilon:%.4f'%(episode,frame_count,c_loss,a_loss,rw,var))
				#print (act)
