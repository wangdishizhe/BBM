from multiprocessing import Pool
import numpy as np
import random
import data_generation_3 as dg
import time
from beta_model import beta_model

N_list = [500,1000]
K = 4
T = 50
seed1 = 9
random.seed(seed1)
seed2 = random.sample(range(100), T)

def write():
    f = open("3_betamodel.txt", "a+")
    if f.tell() == 0:  
        f.write("%s %s %s %s \r\n" % ("N", "K", "P_error", "SEM1"))
    f.write("\n")
    for i in range(4):
        f.write("%.4f " % A1[0, i])
    f.write("\n")
    f.close()

def process_data(param):
    data = dg.data_g(param)
    result = beta_model(data)
    return result  

if __name__ == "__main__":
    time_s = time.time()
    for N in N_list:
        paralist = []
        for t in range(T):
            paralist.append([N, K, seed1, seed2[t]])

        pool = Pool(5)
        A0 = pool.map(process_data, paralist)  
        pool.close()
        pool.join()

        A0 = [np.array([result]) for result in A0] 
        A1 = sum(A0) / len(A0)
        a = []
        for t in range(T):
            a.append(A0[t][0, 2])
        A1[0, 3] = np.std(a) / (T**0.5)
        write()
        pool.close()
        pool.join()
    time_e = time.time()
    print('betamodel',time_e - time_s)
