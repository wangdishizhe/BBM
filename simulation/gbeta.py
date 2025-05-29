import numpy as np
import time
import update_L as uL
from accuracy import compute_best_accuracy
from Initial_for_gbeta import SBM_initial1,SBM_initial4,ER_initial1,ER_initial4
from scipy.special import expit

def initial_check(N,K,A,methods):
    ones = np.ones((N, 1))
    min_L= float("inf") 
    for method in methods:
        Z0,gamma0,eta0=method(N,K,A)
        Theta0 = (np.dot(gamma0, ones.T)+ np.dot(ones, gamma0.T)+ (np.dot(eta0, ones.T) + np.dot(ones, eta0.T)) * np.dot(Z0, Z0.T))
        L0 = uL.Loss(A,Theta0)
        if L0<min_L:
            Z=Z0
            gamma=gamma0
            eta=eta0
            min_L=L0
            Theta=Theta0
    return Z,gamma,eta,min_L,Theta

def gbeta(paralist):
    N = paralist[0]
    K = paralist[1]
    Theta_true = paralist[2]
    A = paralist[3]
    Z_true = paralist[4]
    gamma_true = paralist[5]
    eta_true = paralist[6]
    ss1=paralist[7]
    ss2=paralist[8]


    ones = np.ones((N, 1))
    methods=[SBM_initial1,SBM_initial4,ER_initial1,ER_initial4]
    Z0,gamma0,eta0,L0,Theta0=initial_check(N,K,A,methods)

    s1 = ss1/(N**2)
    s2 = ss2/(N**2)
    s=s1
    ss=s2
    r=0
    epsilon1=1e-10
    while True:
        t=0
        P=expit(Theta0)
        while True:
            eta1 = eta0 + ss* np.reshape(np.sum((A-P)* np.dot(Z0, Z0.T),0),(N, 1))
            Theta01 = (np.dot(gamma0, ones.T)+ np.dot(ones, gamma0.T)+ (np.dot(eta1, ones.T) + np.dot(ones, eta1.T)) * np.dot(Z0, Z0.T))
            L01 = uL.Loss(A,Theta01)  
            if L01 > L0 and t<40:
                ss = ss/2
                t=t+1
            elif t==40:
                eta1=eta0
                L01=L0
                Theta01=Theta0
                break
            else:
                break
        t=0
        P=expit(Theta01)
        while True:
            gamma1 = gamma0 + s* np.reshape(
                np.sum(
                    (
                     A-P
                    ),
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
                gamma1=gamma0
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
            ss=s2
            r += 1
 
    accuracy =  compute_best_accuracy(Z1, Z_true)
    P_hat = expit(Theta1)
    P = expit(Theta_true)
    error_p = np.sum((P_hat - P) ** 2) / np.sum(P**2)
    error_gamma=np.sum((gamma1 - gamma_true) ** 2) / np.sum(gamma_true**2)
    error_eta=np.sum((eta1 - eta_true) ** 2) / np.sum(eta_true**2)
    return (N, K, error_p, 0, 1-accuracy, 0,error_gamma,0, error_eta,0,0,L1)


def gbeta3(paralist):
    N = paralist[0]
    K = paralist[1]
    Theta_true = paralist[2]
    A = paralist[3]
    Z_true = paralist[4]
    gamma_true = paralist[5]
    eta_true = paralist[6]
    ss1=paralist[7]
    ss2=paralist[8]


    ones = np.ones((N, 1))
    methods=[SBM_initial4,ER_initial4]
    Z0,gamma0,eta0,L0,Theta0=initial_check(N,K,A,methods)

    s1 = ss1/(N**2)
    s2 = ss2/(N**2)
    s=s1
    ss=s2
    r=0
    epsilon1=1e-10
    while True:
        t=0
        P=expit(Theta0)
        while True:
            eta1 = eta0 + ss* np.reshape(np.sum((A-P)* np.dot(Z0, Z0.T),0),(N, 1))
            Theta01 = (np.dot(gamma0, ones.T)+ np.dot(ones, gamma0.T)+ (np.dot(eta1, ones.T) + np.dot(ones, eta1.T)) * np.dot(Z0, Z0.T))
            L01 = uL.Loss(A,Theta01)  
            if L01 > L0 and t<40:
                ss = ss/2
                t=t+1
            elif t==40:
                eta1=eta0
                L01=L0
                Theta01=Theta0
                break
            else:
                break
        t=0
        P=expit(Theta01)
        while True:
            gamma1 = gamma0 + s* np.reshape(
                np.sum(
                    (
                     A-P
                    ),
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
                gamma1=gamma0
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
            ss=s2
            r += 1
 
    accuracy =  compute_best_accuracy(Z1, Z_true)
    P_hat = expit(Theta1)
    P = expit(Theta_true)
    error_p = np.sum((P_hat - P) ** 2) / np.sum(P**2)
    error_gamma=np.sum((gamma1 - gamma_true) ** 2) / np.sum(gamma_true**2)
    error_eta=np.sum((eta1 - eta_true) ** 2) / np.sum(eta_true**2)
    return (N, K, error_p, 0, 1-accuracy, 0,error_gamma,0, error_eta,0,0,L1)

def gbeta2(paralist):
    N = paralist[0]
    K = paralist[1]
    P_true = paralist[2]
    A = paralist[3]
    Z_true = paralist[4]
    ss1=paralist[5]
    ss2=paralist[6]


    ones = np.ones((N, 1))
    methods=[SBM_initial1,SBM_initial4,ER_initial1,ER_initial4]
    Z0,gamma0,eta0,L0,Theta0=initial_check(N,K,A,methods)

    s1 = ss1/(N**2)
    s2 = ss2/(N**2)
    s=s1
    ss=s2
    r=0
    epsilon1=1e-10
    while True:
        t=0
        P=expit(Theta0)
        while True:
            eta1 = eta0 + ss* np.reshape(np.sum((A-P)* np.dot(Z0, Z0.T),0),(N, 1))
            Theta01 = (np.dot(gamma0, ones.T)+ np.dot(ones, gamma0.T)+ (np.dot(eta1, ones.T) + np.dot(ones, eta1.T)) * np.dot(Z0, Z0.T))
            L01 = uL.Loss(A,Theta01)  
            if L01 > L0 and t<40:
                ss = ss/2
                t=t+1
            elif t==40:
                eta1=eta0
                L01=L0
                Theta01=Theta0
                break
            else:
                break
        t=0
        P=expit(Theta01)
        while True:
            gamma1 = gamma0 + s* np.reshape(
                np.sum(
                    (
                     A-P
                    ),
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
                gamma1=gamma0
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
            ss=s2
            r += 1
 
    accuracy =  compute_best_accuracy(Z1, Z_true)
    P_hat = expit(Theta1)
    error_p = np.sum((P_hat - P_true) ** 2) / np.sum(P_true**2)
    return (N, K, error_p, 0, 1-accuracy, 0,0,0, 0,0,0,L1)


def gbeta4(paralist):
    N = paralist[0]
    K = paralist[1]
    P_true = paralist[2]
    A = paralist[3]
    Z_true = paralist[4]
    ss1=paralist[5]
    ss2=paralist[6]


    ones = np.ones((N, 1))
    methods=[SBM_initial4,ER_initial4]
    Z0,gamma0,eta0,L0,Theta0=initial_check(N,K,A,methods)

    s1 = ss1/(N**2)
    s2 = ss2/(N**2)
    s=s1
    ss=s2
    r=0
    epsilon1=1e-10
    while True:
        t=0
        P=expit(Theta0)
        while True:
            eta1 = eta0 + ss* np.reshape(np.sum((A-P)* np.dot(Z0, Z0.T),0),(N, 1))
            Theta01 = (np.dot(gamma0, ones.T)+ np.dot(ones, gamma0.T)+ (np.dot(eta1, ones.T) + np.dot(ones, eta1.T)) * np.dot(Z0, Z0.T))
            L01 = uL.Loss(A,Theta01)  
            if L01 > L0 and t<40:
                ss = ss/2
                t=t+1
            elif t==40:
                eta1=eta0
                L01=L0
                Theta01=Theta0
                break
            else:
                break
        t=0
        P=expit(Theta01)
        while True:
            gamma1 = gamma0 + s* np.reshape(
                np.sum(
                    (
                     A-P
                    ),
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
                gamma1=gamma0
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
            ss=s2
            r += 1
 
    accuracy =  compute_best_accuracy(Z1, Z_true)
    P_hat = expit(Theta1)
    error_p = np.sum((P_hat - P_true) ** 2) / np.sum(P_true**2)
    return (N, K, error_p, 0, 1-accuracy, 0,0,0, 0,0,0,L1)