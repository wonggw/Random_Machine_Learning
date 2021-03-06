import gym
import numpy as np
import math
from matplotlib import pyplot as plt
from random import randint
from statistics import median, mean


env = gym.make('CartPole-v1')

award_set =[]

test_run = 15

best_gen =[]

nn_out = []

def lreLu(x):
	alpha=0.2
	return tf.nn.relu(x)-alpha*tf.nn.relu(-x)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def reLu(x):
	return np.maximum(0,x)


def intial_gen(test_run):
	input_weight = []
	input_bias = []

	hidden_weight = []
	out_weight = [] 

	in_node = 4   # 4,2 combination good # 1st layer
	hid_node = 2	# 2nd layer

	for i in range(test_run):
		in_w = np.random.rand(4	,in_node)
		input_weight.append(in_w)

		in_b = np.random.rand((in_node))
		input_bias.append(in_b)

		hid_w = np.random.rand(in_node,hid_node)
		hidden_weight.append(hid_w)

		#print (hid_w)
		out_w = np.random.rand(hid_node, 1)
		out_weight.append(out_w)
	
	generation = [input_weight, input_bias, hidden_weight, out_weight]
	return generation



def nn(obs,in_w,in_b,hid_w,out_w):

	#obs = np.reshape(obs,(1,4))
	obs = np.array(obs).reshape(1,len(obs))
	#obs = obs/np.max(obs) 

	Ain = reLu(np.dot(obs,in_w)+in_b.T)
	#hid_layer = np.dot(Ain,hid_w)
	#Ahid = sigmoid(np.dot(Ain,hid_w))
	Ahid = reLu(np.dot(Ain,hid_w))

	out_put = reLu(np.dot(Ahid,out_w))

	nn_out.append(out_put)
	#print (out_put)
	if out_put  < 0.9002: #(this should be close to < 1(3,2)layers # <0.90001 4,2 layers
		out_put = 0
	else:
		out_put = 1
	

	return out_put

def run_env(env,in_w,in_b,hid_w,out_w):
	obs = env.reset()
	award = 0
	for t in range(999999):
		#env.render() this slows the process
		action = nn(obs,in_w,in_b,hid_w,out_w)
		obs, reward, done, info = env.step(action)
		award += reward 
		if done:
			break
	return award


def rand_run(env,test_run):
	award_set = []
	generations = intial_gen(test_run)

	for episode in range(test_run):# run env 10 time
		in_w  = generations[0][episode]
		in_b = generations[1][episode]
		hid_w =  generations[2][episode]
		out_w =  generations[3][episode]
		award = run_env(env,in_w,in_b,hid_w,out_w)
		award_set = np.append(award_set,award)
	gen_award = [generations, award_set]
	return gen_award  


def mutation(new_dna):

	j = np.random.randint(0,len(new_dna))
	#print (new_dna)
	if ( 0 <j < 10): # controlling rate of amount mutationadam optimizer 
		for i in range(j):
			n = np.random.randint(0,len(new_dna)) #random postion for mutation
			new_dna[n] = new_dna[n] + np.random.rand()

	mut_dna = new_dna

	return mut_dna

def crossover(Dna_list):
	newDNA_list = []
	newDNA_list.append(Dna_list[0])
	newDNA_list.append(Dna_list[1]) 
	
	for l in range(100):  # generation after crassover
		j = np.random.randint(0,len(Dna_list[0]))
		new_dna = np.append(Dna_list[0][:j], Dna_list[1][j:])

		mut_dna = mutation(new_dna)
		newDNA_list.append(mut_dna)

	return newDNA_list


def reproduce(award_set, generations):

	good_award_idx = award_set.argsort()[-2:][::-1] # here only best 2 are selected , [-2:] pick characters from the second-last (included) to the end,  [::-1] make a copy of the same list in reverse order

	good_generation = []
	DNA_list = []

	new_input_weight = []
	new_input_bias = []

	new_hidden_weight = []

	new_output_weight =[]

	new_award_set = []

	for index in good_award_idx:
		
		w1 = generations[0][index]
		#print (w1)
		dna_in_w = w1.reshape(w1.shape[1],-1)
		#print (dna_in_w)
		b1 = generations[1][index]
		dna_b1 = np.append(dna_in_w, b1)

		w2 = generations[2][index]
		dna_whid = w2.reshape(w2.shape[1],-1)
		dna_w2 = np.append(dna_b1,dna_whid)
		#print (dna_w2)
		wh = generations[3][index]
		dna = np.append(dna_w2, wh)

		#print (dna)
		DNA_list.append(dna) # make 2 dna for good gerneration

	newDNA_list = crossover(DNA_list)

	for newdna in newDNA_list: # collection of weights from dna info
		
		newdna_in_w1 = np.array(newdna[:generations[0][0].size]) 
		new_in_w = np.reshape(newdna_in_w1, (-1,generations[0][0].shape[1]))
		new_input_weight.append(new_in_w)

		new_in_b = np.array([newdna[newdna_in_w1.size:newdna_in_w1.size+generations[1][0].size]]).T #bias
		new_input_bias.append(new_in_b)

		sh = newdna_in_w1.size + new_in_b.size
		newdna_in_w2 = np.array([newdna[sh:sh+generations[2][0].size]])
		new_hid_w = np.reshape(newdna_in_w2, (-1,generations[2][0].shape[1]))
		new_hidden_weight.append(new_hid_w)

		sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
		new_out_w   = np.array([newdna[sl:]]).T
		new_output_weight.append(new_out_w)

		new_award = run_env(env, new_in_w, new_in_b, new_hid_w, new_out_w) #bias
		new_award_set = np.append(new_award_set,new_award)

	new_generation = [new_input_weight,new_input_bias,new_hidden_weight,new_output_weight]

	return new_generation, new_award_set


def evolution(env,test_run,n_of_generations):
	gen_award = rand_run(env, test_run)

	current_gens = gen_award[0] 
	current_award_set = gen_award[1]
	best_gen =[]adam optimizer 
	A =[]
	for n in range(n_of_generations):
		new_generation, new_award_set = reproduce(current_award_set, current_gens)
		current_gens = new_generation
		current_award_set = new_award_set
		avg = np.average(current_award_set)
		if avg > 4500:
			best_gen = np.array([current_gens[0][0],current_gens[1][0],current_gens[2][0],current_gens[3][0]])
			np.save("newtest", best_gen)
		a = np.amax(current_award_set)
		print("generation: {}, score: {}".format(n+1, a))
		A = np.append(A, a)
	Best_award = np.amax(A)

	
	plt.plot(A)
	plt.xlabel('generations')
	plt.ylabel('score')

	print('Average accepted score:',mean(A))
	print('Median score for accepted scores:',median(A))
	return plt.show()


n_of_generations = 100000
evolution(env, test_run, n_of_generations)


param = np.load("newtest.npy")


in_w = param[0]
in_b = param[1]
hid_w= param[2]
out_w= param[3]


def test_run_env(env,in_w,in_b,hid_w,out_w):
	obs = env.reset()
	award = 0
	for t in range(5000):
		env.render() #thia slows the process
		action = nn(obs,in_w,in_b,hid_w,out_w)
		obs, reward, done, info = env.step(action)
		award += reward

		print("time: {}, fitness: {}".format(t, award)) 
		if done:
			break
	return award

#print test_run_env(env, in_w, in_b, hid_w,out_w)

