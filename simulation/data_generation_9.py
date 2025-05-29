import numpy as np

def data_g(paralist):
    N = paralist[0]      
    K = paralist[1]          
    seed1 = paralist[2]
    seed2 = paralist[3]
   
    np.random.seed(seed1)
    
    alpha = [1/K] * K
    np.random.seed(seed2)  
    Z = np.random.multinomial(1, alpha, size=N)
    communities = np.argmax(Z, axis=1)  
    
    B = np.eye(K)
    upper_tri_indices = np.triu_indices(K, k=1)
    B[upper_tri_indices] = np.random.uniform(low=0.1, high=0.9, size=len(upper_tri_indices[0]))
    B = np.triu(B) + np.triu(B, k=1).T 
    

    theta = np.zeros(N)
    c0, d0 = 0.7, 0.1
    theta = d0 + (c0 - d0) * np.arange(N) / N
    
    p = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            p[i, j] = theta[i] * theta[j] * B[communities[i], communities[j]]
    
    np.random.seed(seed2)
    A_upper = np.triu(np.random.binomial(1, p), k=1)
    A = A_upper + A_upper.T

    return (N, K, p, A, Z)
