from multiprocessing import Pool
import numpy as np
import random
import data_generation_6 as dg
from gbeta import gbeta3
import time

N_list = [500, 1000]
K = 4
T = 50
seed1 = 17
random.seed(seed1)
seed2 = random.sample(range(100), T)

step_sizes = {
    500: (4096, 128),
    1000: (4096, 512)
}

def process_data(param):
    data = dg.data_g(param)
    N = param[0]  
    step_size = step_sizes[N]  
    data_s = data + step_size
    result = gbeta3(data_s)
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

        a = [A0[t][0, 2] for t in range(T)]
        A1[0, 3] = np.std(a) / (T**0.5)

        b = [A0[t][0, 4] for t in range(T)]
        A1[0, 5] = np.std(b) / (T**0.5)

        c = [A0[t][0, 6] for t in range(T)]
        A1[0, 7] = np.std(c) / (T**0.5)

        d = [A0[t][0, 8] for t in range(T)]
        A1[0, 9] = np.std(d) / (T**0.5)
        
        with open(f"6_gbeta.txt", "a+") as f:
            if f.tell() == 0: 
                f.write(
                    "%s %s %s %s %s %s %s %s %s %s \r\n"
                    % ("N", "K", "P_error", "SEM1", "Detection_error", "SEM2", "gamma_error", "SEM3", "eta_error", "SEM4")
                )
            f.write("\n")
            for i in range(10):
                f.write("%.4f " % A1[0, i])
            f.write("\n")
                    
    time_e = time.time()
    print('gbeta', time_e - time_s)