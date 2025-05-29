import numpy as np
from scipy.special import expit

def beta_model(paralist):
    N = paralist[0]
    K = paralist[1]
    Theta_true = paralist[2]
    A = paralist[3]
    tol=1e-3
    degree_sequence=np.sum(A,axis=1)
    n = len(degree_sequence)
    beta = np.zeros((n,1))
    for iteration in range(1000):
        beta_old = beta.copy()
        for i in range(n):
            eps=1e-10
            sum_terms = sum(1 / (np.exp(-beta[j]+eps) + np.exp(beta[i]+eps)) for j in range(n) if j != i)
            beta[i] = np.log(degree_sequence[i]+eps) - np.log(sum_terms+eps)
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    ones = np.ones((n, 1))
    Theta=np.dot(beta,ones.T)+np.dot(ones,beta.T)
    P_hat=expit(Theta)
    P= expit(Theta_true)
    error_theta = np.sum(((P_hat - P) ) ** 2) / np.sum(P**2)
    return (N, K,error_theta,0)

def beta_model2(paralist):
    N = paralist[0]
    K = paralist[1]
    P = paralist[2]
    A = paralist[3]
    tol=1e-3
    degree_sequence=np.sum(A,axis=1)
    n = len(degree_sequence)
    beta = np.zeros((n,1))
    for iteration in range(1000):
        beta_old = beta.copy()
        for i in range(n):
            eps=1e-10
            sum_terms = sum(1 / (np.exp(-beta[j]+eps) + np.exp(beta[i]+eps)) for j in range(n) if j != i)
            beta[i] = np.log(degree_sequence[i]+eps) - np.log(sum_terms+eps)
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    ones = np.ones((n, 1))
    Theta=np.dot(beta,ones.T)+np.dot(ones,beta.T)
    P_hat=expit(Theta)
    error_theta = np.sum(((P_hat - P) ) ** 2) / np.sum(P**2)
    return (N, K,error_theta,0)
