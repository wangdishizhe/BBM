import numpy as np
import gc
from sklearn.cluster import KMeans
from accuracy import compute_best_accuracy
from scipy.special import expit

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


def DCSBM_score(A,K):
    N = A.shape[0]

    Z=score(A,K)
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  

    return Z0

