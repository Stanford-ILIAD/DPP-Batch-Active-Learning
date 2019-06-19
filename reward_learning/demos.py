from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo
import sys

def batch(task, method, N, M, b):
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    B = 20*b

    simulation_object = create_env(task)
    d = simulation_object.num_of_features
	
    w_true = 2*np.random.rand(d)-1
    w_true = w_true / np.linalg.norm(w_true)
    print('If in automated mode: true w = {}'.format(w_true/np.linalg.norm(w_true)))
	
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    i = 0
    while i < N:
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('Samples so far: ' + str(i))
        print('w estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        print('Alignment = {}'.format(mean_w_samples.dot(w_true)/np.linalg.norm(mean_w_samples)))
        inputA_set, inputB_set = run_algo(method, simulation_object, w_samples, b, B)
        for j in range(b):
            input_A = inputA_set[j]
            input_B = inputB_set[j]
            psi, s = get_feedback(simulation_object, input_B, input_A, w_true)
            psi_set.append(psi)
            s_set.append(s)
        i += b
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    mean_w_samples = np.mean(w_samples, axis=0)
    print('Samples so far: ' + str(N))
    print('w estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
    print('Alignment = {}'.format(mean_w_samples.dot(w_true)/np.linalg.norm(mean_w_samples)))



def nonbatch(task, method, N, M):
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
	
    w_true = 2*np.random.rand(d)-1
    w_true = w_true / np.linalg.norm(w_true)
    print('If in automated mode: true w = {}'.format(w_true/np.linalg.norm(w_true)))
	
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    for i in range(N):
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('Samples so far: ' + str(i))
        print('w estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        print('Alignment = {}'.format(mean_w_samples.dot(w_true)/np.linalg.norm(mean_w_samples)))
        input_A, input_B = run_algo(method, simulation_object, w_samples)
        psi, s = get_feedback(simulation_object, input_A, input_B, w_true)
        psi_set.append(psi)
        s_set.append(s)
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    print('Samples so far: ' + str(N))
    print('w estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
    print('Alignment = {}'.format(mean_w_samples.dot(w_true)/np.linalg.norm(mean_w_samples)))


