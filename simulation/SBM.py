import numpy as np
import gc
from sklearn.cluster import KMeans
from accuracy import  compute_best_accuracy
from scipy.special import expit

def SBM(paralist):
    N = paralist[0]
    K = paralist[1]
    Theta_true = paralist[2]
    A = paralist[3]
    Z_true = paralist[4]


    eigvals, eigvecs = np.linalg.eigh(A)
    sorted_indices = np.argsort(np.abs(eigvals))[::-1]  
    eigvecs_selected = eigvecs[:, sorted_indices[:K]] 

    kmeans = KMeans(n_clusters=K,n_init='auto',random_state=1)

    Z = kmeans.fit_predict(eigvecs_selected)
    del kmeans  
    gc.collect() 
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  
    accuracy=compute_best_accuracy(Z0,Z_true)
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
    P_hat=Z0 @ B_hat @ Z0.T
    P =  expit(Theta_true)
    error_theta = np.sum(((P_hat - P) ) ** 2) / np.sum(P**2)

    return (N, K,error_theta,0,1-accuracy,0)

def SBM2(paralist):
    N = paralist[0]
    K = paralist[1]
    P = paralist[2]
    A = paralist[3]
    Z_true = paralist[4]


    eigvals, eigvecs = np.linalg.eigh(A)
    sorted_indices = np.argsort(np.abs(eigvals))[::-1]  
    eigvecs_selected = eigvecs[:, sorted_indices[:K]] 

    kmeans = KMeans(n_clusters=K,n_init='auto',random_state=1)

    Z = kmeans.fit_predict(eigvecs_selected)
    del kmeans  
    gc.collect() 
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  
    accuracy=compute_best_accuracy(Z0,Z_true)
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
    P_hat=Z0 @ B_hat @ Z0.T
    error_theta = np.sum(((P_hat - P) ) ** 2) / np.sum(P**2)

    return (N, K,error_theta,0,1-accuracy,0)


    

