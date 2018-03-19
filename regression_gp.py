#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt



def se_covariance(x1, x2, param):
    sqdist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1, x2.T)
    return np.exp(-.5 * (1/param) * sqdist)



def generate_data(n, n_s):
    x = (np.linspace(0, 4*np.pi, n) + 0.01*np.random.randn(n)).reshape(-1,1)
    #x = np.sort(np.random.uniform(0, 4*np.pi, n)).reshape(-1,1)
    x_s = np.linspace(0, 6*np.pi, n_s).reshape(-1,1)
    y = np.sin(x) + (4.0/(25*np.pi**2)) * x**2 - 8/(5*np.pi)*x + 4
    y_test = np.sin(x_s) + (4.0/(25*np.pi**2)) * x_s**2 - 8/(5*np.pi)*x_s + 4
    return x, x_s, y, y_test

if __name__ == "__main__":
    np.random.seed(10000)
    n = 10
    n_s = 50
    l = 1.2
    
    #generating train and test input, and train output
    x, x_s, y, y_test = generate_data(n, n_s)

    #PRIOR
    #define the covariance matrix from this set of training data
    K_ss = se_covariance(x_s, x_s, l)
    #now want to find the square root of this matrix
    #Cholesky method will approximate it
    L_ss = np.linalg.cholesky(K_ss + 1e-5 * np.eye(n_s))
    #generate 3 functions
    f_prior = np.dot(L_ss, np.random.randn(n_s, 3))
    plt.figure()
    plt.plot(x_s, f_prior)
    plt.title('Samples generated from the prior')
    plt.show()

    #now want to condition the prior on the test inputs to generate
    #posterior distribution
    K = se_covariance(x, x, l)
    K_s = se_covariance(x, x_s, l)
    
    #creating mean of posterior
    mu_s = np.matmul(np.matmul(K_s.T, np.linalg.inv(K)), y)
    sigma_s = K_ss - np.matmul(np.matmul(K_s.T, np.linalg.inv(K)), K_s)
    std = np.sqrt(sigma_s[np.eye(sigma_s.shape[0]).astype(np.bool)])

    #now sample from this distribution
    L_ss = np.linalg.cholesky(sigma_s + 1e-5*np.eye(sigma_s.shape[0]))
    y_s = np.dot(L_ss, np.random.randn(n_s,1)) + mu_s
    plt.figure()
    plt.plot(x_s, y_s, 'r', label='Posterior')
    plt.plot(x_s, y_test, 'b', label='True')
    plt.legend()
    plt.gca().fill_between(x_s.flat,
                           mu_s.ravel() - 2 * std,
                           mu_s.ravel() + 2 * std,
                           color="#dddddd")
    plt.scatter(x,y)
    plt.title('Samples generated from the posterior')
    plt.show()


    #save the data
    np.save('x', x)
    np.save('y', y)
    np.save('x_s', x_s)
    np.save('y_test', y_test)
    np.save('y_gp', y_s)
    np.save('mu_s', mu_s)
    np.save('std', std)
