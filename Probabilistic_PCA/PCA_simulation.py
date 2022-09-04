#!/home/a.ghaderi/.conda/envs/envjm/bin/python
import os
import numpy as np
from scipy import stats
from time import time
import matplotlib.pyplot as plt

from numba import njit
import tensorflow as tf

import sys
sys.path.append('../../')
from bayesflow.networks import InvertibleNetwork, InvariantNetwork
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.trainers import ParameterEstimationTrainer
from bayesflow.diagnostics import *
from bayesflow.models import GenerativeModel


def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------

    Arguments:
    batch_size : int -- the number of samples to draw from the prior
    ----------

    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
    """

    # Prior ranges for the simulator
    # w1:5 ~ U(-1.0, 1.0)
    # psi1:4 ~ U(-1.0, 1.0)
    n_parameters = 9
    p_samples = np.random.uniform(low=(-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
                                  high=(1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0, 1.0), size=(batch_size, n_parameters))
    return p_samples.astype(np.float32)

def PCA_trial(params, n_trials, P = 4, D = 2):
    """Simulates a trial from the pca."""
    
    # number of non-zero loadings
    M = int(D*(P-D)+ D*(D-1)/2)
    # lower diagonal elements of L
    L = params[0:M]
    # vector of variances
    psi = params[M:M+P]
        
    # the loading matrix
    idx = 0
    L = np.zeros((P, D))
    for j in range(D):
        L[j,j] = 1
        for i in range(j+1,P):
            L[i,j] = L_t[idx]
            idx = idx + 1;

    # the latent variable
    z = np.random.multivariate_normal(np.repeat(0, D), np.identity(D), n_trials)
    # the error
    epsilon = np.random.multivariate_normal(np.repeat(0, P), np.diag(psi), n_trials)

    # simulated data via multivariate normal
    y = np.dot(z, np.transpose(L)) +  epsilon;

    return y

def batch_simulator(prior_samples, n_obs, D = 2, P = 4):
    """
    Simulate multiple diffusion_model_datasets.
    """

    n_sim = prior_samples.shape[0]
    sim = np.empty((n_sim, n_obs, P), dtype=np.float32)

    # Simulate diffusion data
    for i in range(n_sim):
        sim[i] = PCA_trial(prior_samples[i], n_obs)

    return sim


# Connect the networks through a SingleModelAmortizer instance.
summary_net = InvariantNetwork()
inference_net = InvertibleNetwork({'n_params': 9})
amortizer = SingleModelAmortizer(inference_net, summary_net)

# Connect the prior and simulator through a GenerativeModel class which will take care of forward inference.
generative_model = GenerativeModel(prior, batch_simulator)

trainer = ParameterEstimationTrainer(
    network=amortizer,
    generative_model=generative_model,
    checkpoint_path="checkpoint"
)


# Variable n_trials
def prior_N(n_min=60, n_max=300):
    """
    A prior or the number of observation (will be called internally at each backprop step).
    """

    return np.random.randint(n_min, n_max + 1)


# Experience-replay training
losses = trainer.train_experience_replay(epochs=500,
                                         batch_size=32,
                                         iterations_per_epoch=500,
                                         capacity=100,
                                         n_obs=prior_N)
 
# Validate (quick and dirty)
n_param_sets = 300
n_samples = 300
n_trials = 1000

true_params = prior(n_param_sets)
x = batch_simulator(true_params, n_trials).astype(np.float32)
param_samples = amortizer.sample(x, n_samples=n_samples)
param_means = param_samples.mean(axis=0)
true_vs_estimated(true_params, param_means, ['L1', 'L2', 'L3', 'L4', 'L5', 'psi1', 'psi2', 'psi3', 'psi4'], filename="PCA_simulation")
