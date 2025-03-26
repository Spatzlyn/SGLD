import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal

#N(mu,cov)-> target distribution
mu=np.array([1.0,1.0])
cov=np.array([[0.5,0.0],[0.0,0.5]])
cov_inverse=np.linalg.inv(cov)

#logprob: log probability function, gradlog: gradient of log probability
def logprob(theta):
    diff=theta-mu
    return -0.5*diff.T @ cov_inverse @ diff

def gradlog(theta):
    return -cov_inverse@(theta-mu)