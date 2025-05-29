import numpy as np
from scipy.special import expit

def data_g(paralist):
    N = paralist[0]
    K = paralist[1]
    seed1 = paralist[2]
    seed2 = paralist[3]

    np.random.seed(seed1)
    
    beta=np.random.normal(-3, 1, N).reshape(N, 1)
    eta= np.ones((N,1))*2
    gamma=beta-eta
    alpha = [1 / K] * K
    Z = np.random.multinomial(1, alpha, N)
    ones = np.ones((N, 1))
    Theta = (
        np.dot(gamma, ones.T)
        + np.dot(ones, gamma.T)
        + (np.dot(eta, ones.T) + np.dot(ones, eta.T)) * np.dot(Z, Z.T)
    )
    P = expit(Theta)
    np.random.seed(seed2)
    A_upper = np.triu(np.random.binomial(1, P), k=0)
    A = A_upper + A_upper.T - np.diag(np.diag(A_upper))

    return (N, K, Theta, A, Z,gamma,eta)

