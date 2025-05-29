import numpy as np

def Loss(A,theta):
    expt=np.exp(theta)
    term1 = A*theta
    term2 = np.log(1 + expt)
    loss = - np.sum(term1 - term2)
    return loss

def uL(L,A,gamma, eta, Z):
    N, K = np.shape(Z)  # 节点数和社区数

    L_add = np.zeros((N, K))
    X = np.ones((N, K, N))
    Y = np.zeros((N, K, N))
    Iden = np.identity(N)
    ones = np.ones((N, 1))
    term1=np.dot(gamma, ones.T) + np.dot(ones, gamma.T)
    term2=np.dot(eta, ones.T) + np.dot(ones, eta.T)
    for i in range(N):
        X[i, :, i] = 0
        Y[:, :, i] = Z * X[:, :, i] 

    for k in range(K):
        Zero = np.zeros((N, K))
        Zero[:, k] = 1 
        M = np.einsum('ij,jk->jki', Iden, Zero)
        Z_tensor = Y + M
        einsum=np.einsum('ij,kji->ik', Zero, Z_tensor)
        Theta_add = term1 + term2 * einsum
        expt_add=np.exp(Theta_add)
        term1_add = A * Theta_add
        term2_add = np.log(1 + expt_add)
        L_add[:, k] = - np.sum(term1_add - term2_add, 1)
    
    Z1 = np.zeros((N, K))
    best_k = np.argmin(L_add, axis=1)  
    Z1[np.arange(N), best_k] = 1 
    Theta_new = term1 + term2 * np.dot(Z1,Z1.T)
    Loss_new=Loss(A,Theta_new)
        
    if Loss_new >= L:
        diff_indices = np.where(~np.all(Z == Z1, axis=1))[0]
        if diff_indices.size > 0:
            i_diff = diff_indices[0] 
            Z2 = Z.copy()
            Z2[i_diff, :] = 0 
            trial_losses = np.zeros(K)
            for k_ in range(K):
                Z2[i_diff, k_] = 1 
                Theta_trial = compute_Theta(gamma, eta, Z2)
                trial_losses[k_] = Loss(A, Theta_trial)
                Z2[i_diff, k_] = 0  
            best_k = np.argmin(trial_losses)
            Z2[i_diff, best_k] = 1  
            Z1 = Z2  
        else:
            Z1 = Z.copy()
    return Z1


def compute_Theta(gamma, eta, Z):
    N, K = Z.shape
    onesN = np.ones((N, 1))
    part1 = np.dot(gamma, onesN.T) + np.dot(onesN, gamma.T)
    part2 = np.dot(eta, onesN.T) + np.dot(onesN, eta.T)
    ZZt   = np.dot(Z, Z.T)
    Theta = part1 + part2 * ZZt
    return Theta