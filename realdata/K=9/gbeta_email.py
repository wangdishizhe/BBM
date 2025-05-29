import numpy as np
import update_L as uL
from Initial_for_gbeta import SBM_initial1,SBM_initial4
from scipy.special import expit


def gbeta(A,K,s1,s2):
    N=A.shape[0]
    ones = np.ones((N, 1))
    Z0,gamma0,eta0=SBM_initial4(N,K,A)
    Theta0=(np.dot(gamma0, ones.T)+ np.dot(ones, gamma0.T)+ (np.dot(eta0, ones.T) + np.dot(ones, eta0.T)) * np.dot(Z0, Z0.T))
    L0=uL.Loss(A,Theta0)
    s1=s1/N**2
    s2=s2/N**2
    s = s1
    ss= s2
    r=0
    epsilon1=1e-10
    while True:
        t=0
        P=expit(Theta0)
        while True:
            gamma1 = gamma0 + ss* np.reshape(np.sum((A-P),0),(N, 1))
            Theta01 = (np.dot(gamma1, ones.T)+ np.dot(ones, gamma1.T)+ (np.dot(eta0, ones.T) + np.dot(ones, eta0.T)) * np.dot(Z0, Z0.T))
            L01 = uL.Loss(A,Theta01)  
            if L01 > L0 and t<40:
                ss = ss/2
                t=t+1
            elif t==40:
                gamma1=gamma0
                L01=L0
                Theta01=Theta0
                break
            else:
                break
        t=0
        P=expit(Theta01)
        while True:
            eta1 = eta0 + s* np.reshape(
                np.sum(
                    (
                     A-P
                    )* np.dot(Z0, Z0.T),
                    0,
                ),
                (N, 1),
            )
            Theta02 = (np.dot(gamma1, ones.T)+ np.dot(ones, gamma1.T)+ (np.dot(eta1, ones.T) + np.dot(ones, eta1.T)) * np.dot(Z0, Z0.T))
            L02 = uL.Loss(A,Theta02)  
            if L02 > L01 and t<40:
                s = s/2
                t=t+1
            elif t==40:
                eta1=eta0
                L02=L01
                break
            else:
                break
        
        Z1=uL.uL(L02,A, gamma1, eta1, Z0)

        Theta1 = (
            np.dot(gamma1, ones.T)
            + np.dot(ones, gamma1.T)
            + (np.dot(eta1, ones.T) + np.dot(ones, eta1.T)) * np.dot(Z1, Z1.T)
        )
        L1 = uL.Loss(A, Theta1)
        
        error = abs(L0 - L1) / abs(L0)
        if error < epsilon1 or r>100:
            break
        else:
            gamma0 = gamma1.copy()
            eta0 = eta1.copy()
            Z0 = Z1.copy()
            L0 = L1.copy()
            s=s1
            ss=s1
            r += 1

    return Z1,gamma1,eta1+gamma1
