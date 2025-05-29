import numpy as np
import gc
from sklearn.cluster import KMeans
from scipy.special import expit

def SBM(A,K):
    N=A.shape[0]
    eigvals, eigvecs = np.linalg.eigh(A)
    sorted_indices = np.argsort(np.abs(eigvals))[::-1]  
    eigvecs_selected = eigvecs[:, sorted_indices[:K]] 

    kmeans = KMeans(n_clusters=K,n_init='auto',random_state=1)

    Z = kmeans.fit_predict(eigvecs_selected)
    del kmeans  
    gc.collect() 
    Z0 = np.zeros((N, K))  
    Z0[np.arange(N), Z] = 1  


    return Z0


