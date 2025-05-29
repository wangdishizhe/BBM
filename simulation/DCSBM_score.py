import numpy as np
import gc
from sklearn.cluster import KMeans
from accuracy import compute_best_accuracy
from scipy.special import expit

def P_for_DCSBM(A, Z):
    N, K = Z.shape  

    k = A.sum(axis=1)

    theta = np.zeros(N)
    for r in range(K):
        idx = np.where(Z[:, r] == 1)[0]
        total_degree = k[idx].sum()
        if total_degree > 0:
            theta[idx] = k[idx] / total_degree
        else:
            theta[idx] = 0

    B_hat = np.zeros((K, K))
    for r in range(K):
        idx_r = np.where(Z[:, r] == 1)[0]
        for s in range(K):
            idx_s = np.where(Z[:, s] == 1)[0]
            if r == s:
                block = A[np.ix_(idx_r, idx_r)]
                off_diag = np.sum(block) - np.sum(np.diag(block))
                total_edges = off_diag / 2 + np.sum(np.diag(block))
            else:
                total_edges = np.sum(A[np.ix_(idx_r, idx_s)])
            denom = np.sum(np.outer(theta[idx_r], theta[idx_s]))
            B_hat[r, s] = total_edges / denom if denom > 0 else 0
    
    P_hat = np.outer(theta, theta) * (Z @ B_hat @ Z.T)
    
    return P_hat

def score(A, K):
    A = A.copy()
    n = A.shape[0]
    w, v = np.linalg.eigh(A)
       
    idx = np.argsort(np.abs(w))[::-1]  
    v_k = v[:, idx[:K]]     
    eps=1e-9
    base_vector = v_k[:, 0:1] 
    R = v_k[:, 1:] / (base_vector+eps)
    Tn = np.log(n)
    R_star = np.clip(R, -Tn, Tn)
    kmeans = KMeans(n_clusters=K,n_init='auto',random_state=1)
    Z = kmeans.fit_predict(R_star)
    del kmeans
    gc.collect()

    return Z


def DCSBM_score(paralist):
    N = paralist[0]
    K = paralist[1]
    Theta_true = paralist[2]
    A = paralist[3]
    Z_true = paralist[4]

    Z=score(A,K)
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  
    accuracy = compute_best_accuracy(Z0, Z_true)
    P_hat=P_for_DCSBM(A,Z0)
    P =  expit(Theta_true)
    P_error = np.sum(((P_hat - P)) ** 2) / np.sum(P**2)

    return (N, K, P_error, 0, 1-accuracy, 0)


def DCSBM_score2(paralist):
    N = paralist[0]
    K = paralist[1]
    P = paralist[2]
    A = paralist[3]
    Z_true = paralist[4]

    Z=score(A,K)
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  
    accuracy = compute_best_accuracy(Z0, Z_true)
    P_hat=P_for_DCSBM(A,Z0)
    P_error = np.sum(((P_hat - P)) ** 2) / np.sum(P**2)

    return (N, K, P_error, 0, 1-accuracy, 0)
