import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import gc
import time
from DCSBM_score import score
from sklearn_extra.cluster import KMedoids
from scipy.special import expit

def loss_function_SBM(params, B_hat, Z, N, K):
    gamma = params[:K].reshape(K, 1)
    eta = params[K:].reshape(K, 1)

    ones_k=np.ones((K, 1))
    gamma_term = np.dot(gamma, ones_k.T) + np.dot(ones_k, gamma.T)
    eta_term = (np.dot(eta, ones_k.T) + np.dot(ones_k, eta.T)) * np.eye(K)
    theta = gamma_term + eta_term
    p_est = expit(theta)
    loss = np.sum((B_hat - p_est) ** 2)
    return loss

def estimate_gamma_eta_SBM(B_hat, Z, N, K):
    initial_guess = np.zeros(2 * K)
    result = minimize(loss_function_SBM, initial_guess, args=(B_hat,Z, N, K), method='BFGS')
    gamma_est = result.x[:K].reshape(K, 1)
    eta_est = result.x[K:].reshape(K, 1)
    return gamma_est, eta_est

def ER_initial4(N,K,sn,A):
    p=np.sum(A)/(N**2)
    gamma0=np.zeros((N,1))
    epsilon=1e-10
    Z=score(A,K)
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  
    p_safe = np.clip(p, epsilon, sn - epsilon)
    gamma0 = np.log(p_safe / (sn - p_safe)) / 2
    eta0=np.zeros((N,1))
    return Z0,gamma0,eta0


def SBM_initial4(N,K,A):
    Z=score(A,K)
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  
    B_hat = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            nodes_k1 = np.where(Z == k1)[0]
            nodes_k2 = np.where(Z == k2)[0]
            if k1 == k2:
                block = A[np.ix_(nodes_k1, nodes_k1)]
                n_k = len(nodes_k1)
                off_diag = np.sum(block) - np.sum(np.diag(block))
                total_edges = off_diag / 2 + np.sum(np.diag(block)) 
                num_pairs = n_k * (n_k + 1) / 2
            else:
                total_edges = np.sum(A[np.ix_(nodes_k1, nodes_k2)])
                num_pairs = len(nodes_k1) * len(nodes_k2)
            B_hat[k1, k2] = total_edges / num_pairs if num_pairs > 0 else 0.0
    #print(B_hat,sn)
    gamma0_K, eta0_K = estimate_gamma_eta_SBM(B_hat, Z0, N, K)

    gamma0_N = gamma0_K[Z] 
    eta0_N = eta0_K[Z]  

    return Z0, gamma0_N, eta0_N

def SBM_initial_K_2(N,K,sn,A):
    Z=score(A,K)
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  
    B_hat = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            nodes_k1 = np.where(Z == k1)[0]
            nodes_k2 = np.where(Z == k2)[0]
            if k1 == k2:
                block = A[np.ix_(nodes_k1, nodes_k1)]
                n_k = len(nodes_k1)
                off_diag = np.sum(block) - np.sum(np.diag(block))
                total_edges = off_diag / 2 + np.sum(np.diag(block)) 
                num_pairs = n_k * (n_k + 1) / 2
            else:
                total_edges = np.sum(A[np.ix_(nodes_k1, nodes_k2)])
                num_pairs = len(nodes_k1) * len(nodes_k2)
            B_hat[k1, k2] = total_edges / num_pairs if num_pairs > 0 else 0.0
    print(B_hat,sn)
    theta_hat=np.log((B_hat/sn)/(1-(B_hat/sn)))

    print(theta_hat)
    gamma=theta_hat[0,1]/2
    eta1=theta_hat[0,0]/2-gamma
    eta2=theta_hat[1,1]/2-gamma
    gamma0_N = (np.ones(N) * gamma).reshape((N, 1))
    eta0_N = np.where(Z == 0, eta1, eta2).reshape((N, 1))
    print(gamma0_N)
    print(eta0_N)
    
    return Z0, gamma0_N, eta0_N

def SBM_initial1(N,K,sn,A):

    eigvals, eigvecs = np.linalg.eigh(A)
    sorted_indices = np.argsort(np.abs(eigvals))[::-1]
    eigvecs_selected = eigvecs[:, sorted_indices[:K]]

    kmeans = KMeans(n_clusters=K,n_init='auto',random_state=1)
    Z = kmeans.fit_predict(eigvecs_selected)
    del kmeans
    gc.collect()

    Z0 = np.zeros((N, K))
    Z0[np.arange(N), Z] = 1  

    B_hat = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            nodes_k1 = np.where(Z == k1)[0]
            nodes_k2 = np.where(Z == k2)[0]
            if k1 == k2:
                block = A[np.ix_(nodes_k1, nodes_k1)]
                n_k = len(nodes_k1)
                off_diag = np.sum(block) - np.sum(np.diag(block))
                total_edges = off_diag / 2 + np.sum(np.diag(block)) 
                num_pairs = n_k * (n_k + 1) / 2
            else:
                total_edges = np.sum(A[np.ix_(nodes_k1, nodes_k2)])
                num_pairs = len(nodes_k1) * len(nodes_k2)
            B_hat[k1, k2] = total_edges / num_pairs if num_pairs > 0 else 0.0
    #print(B_hat)
    gamma0_K, eta0_K = estimate_gamma_eta_SBM(B_hat, sn, Z0, N, K)

    gamma0_N = gamma0_K[Z] 
    eta0_N = eta0_K[Z]  

    return Z0, gamma0_N, eta0_N