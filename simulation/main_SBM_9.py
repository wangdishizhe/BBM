from multiprocessing import Pool
import numpy as np
import random
import data_generation_9 as dg
from SBM import SBM2
import time

N_list = [500,1000]
K = 4
T = 50
seed1 = 0
random.seed(seed1)
seed2 = random.sample(range(100), T)


def write():
    f = open(f"9_SBM.txt", "a+")
    if f.tell() == 0: 
        f.write(
            "%s %s %s %s %s %s \r\n"
            % ("N", "K",  "P_error", "SEM1", "Detection_error", "SEM2")
        )
    f.write("\n")
    for i in range(6):
        f.write("%.4f " % A1[0, i])
    f.write("\n")  
    f.close()
 
def process_data(param):
    data = dg.data_g(param)
    result = SBM2(data)
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
        b = []
        for t in range(T):
            b.append(A0[t][0, 4])
        A1[0, 5] = np.std(b) / (T**0.5)
        write()
        pool.close()
        pool.join()
    time_e = time.time()
    print('SBM',time_e - time_s)

