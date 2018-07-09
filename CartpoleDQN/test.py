import gym
import random
import numpy as np
import tensorflow as tf
import model as M
from collections import deque 

class memory():
	def __init__(self):
		self.data = deque()
	def push(self,data):
		self.data.append(data)
		if len(self.data)>10000:
			self.data.popleft()
	def next_batch(self,BSIZE):
		return random.sample(self.data,BSIZE)


def main_structure(inp_holder,scope):
	with tf.variable_scope(scope):

		mod = M.Model(inp_holder,[None,4])
		mod.fcLayer(32,activation=M.PARAM_LRELU)
		#mod.dropout(0.8)
		mod.fcLayer(64,activation=M.PARAM_LRELU)
		mod.fcLayer(128,activation=M.PARAM_LRELU)
		mod.fcLayer(2)
		return mod.get_current_layer()

def build_graph():
	envHolder = tf.placeholder(tf.float32,[None,4])
	actionHolder = tf.placeholder(tf.float32,[None,2])
	scoreHolder = tf.placeholder(tf.float32,[None])

	output = main_structure(envHolder,'output')
	# action_out have the same value as next_score
	action_out = tf.reduce_sum(output*actionHolder,axis=1)
	action = tf.argmax(output,axis=1)
	next_score = tf.reduce_max(output,axis=1)

	loss = tf.reduce_mean(tf.square(scoreHolder - action_out))

	train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

	var_o = M.get_trainable_vars('output')

	return output,envHolder,actionHolder,scoreHolder,action,action_out,next_score,train_step,loss,var_o

# Constants
EXPLORE = 10000
TRAIN = 20000
BSIZE = 32
# Explore and training
frame_count = 0
episode = 0
GAMMA = 0.99
EPS=0.8

epsilon = EPS

output,envHolder,actionHolder,scoreHolder,action,action_out,next_score,train_step,loss,var_o = build_graph()

env = gym.make('CartPole-v1')
observation = env.reset()

memory = memory()

with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	saver = tf.train.Saver()
	frame_count = 0
	env0 =[]
	rw = 0
	act = 0
	observation, reward, done, info = env.step(act)
	env0 = observation

	while True:
		#print (reward)
		action_array = np.zeros(2)
		if random.random()<epsilon:
			act = np.random.randint(2, size=1)
			action_array[act] = 1
			#act = action_array[0]
		else:
			out,act,n_score= sess.run([output,action,next_score],feed_dict={envHolder:[env0]})
			act = act[0]
			n_score = n_score[0]
			#convert it into 1 hot
			action_array[act] = 1
			if frame_count%200==0:
				print('Episode:%d\tFrame:%d\tReward:%d\tQ:%.4f\tAction:%d'%(episode,frame_count,rw,n_score,act))
				print(epsilon)

		observation, reward, done, info = env.step(int(act))		
		env1 = observation
		memory.push([env0,env1,action_array,reward,done])

		if done:
			env.reset()
			observation, reward, done, info = env.step(0)
			env0 = [observation[0],observation[1],observation[2],observation[3]] *1
			episode += 1
			rw = 0
		else:
			env0 = env1
			rw += 1
			
		frame_count += 1

		if frame_count>14000:
			env.render()
		# training
		if frame_count>=EXPLORE:
			train_batch = memory.next_batch(BSIZE)
			s0_batch = [i[0] for i in train_batch]
			a_batch = [i[2] for i in train_batch]
			rw_batch = []
			for i in train_batch:
				if i[4]:
					rw_batch.append(i[3])
				else:
					scr_next = sess.run(next_score,feed_dict={envHolder:[i[1]]})[0]
					rw_batch.append(i[3]+GAMMA*scr_next)
			_,ls = sess.run([train_step,loss],feed_dict={envHolder:s0_batch, actionHolder:a_batch, scoreHolder:rw_batch})
	
			epsilon -= EPS/(TRAIN-EXPLORE)
			epsilon = max(0.,epsilon)

			if frame_count%100000==0:
				saver.save(sess,'./model/'+str(frame_count)+'.ckpt')
				print('Episode:%.4f\tFrame:%d\tLoss:%.4f\tEpsilon:%.4f\tReward:%.4f'%(episode,frame_count,ls,epsilon,rw))
