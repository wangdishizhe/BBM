import numpy as np


def compute_best_accuracy(Z_hat, Z_true):


    labels_hat = np.argmax(Z_hat, axis=1)
    labels_true = np.argmax(Z_true, axis=1)
  
    n = len(labels_hat)  

    A_true = np.equal.outer(labels_true, labels_true)  
    A_hat  = np.equal.outer(labels_hat, labels_hat)    


    mismatch_indicator = ((A_true.astype(int) + A_hat.astype(int)) == 1)

    error_count = np.sum(np.triu(mismatch_indicator, k=1))
    
    clustering_error = (2 / (n * (n - 1))) * error_count

    return 1 - clustering_error
