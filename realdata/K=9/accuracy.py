from itertools import permutations
import numpy as np


def compute_best_accuracy_2(Z_hat, Z_true):
    # Number of clusters
    k = Z_hat.shape[1]
    #Z_ud = np.zeros_like(Z_hat)
    best_accuracy = 0

    for perm in permutations(range(k)):
        # Permute the columns of Z_hat according to the current permutation
        Z_hat_permuted = Z_hat[:, perm]

        correct_assignments = np.sum(np.argmax(Z_hat_permuted, axis=1) == np.argmax(Z_true, axis=1))
        accuracy = correct_assignments / Z_hat.shape[0]

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            #Z_ud = Z_hat_permuted

    return best_accuracy


def compute_best_accuracy(Z_hat, Z_true):


    labels_hat = np.argmax(Z_hat, axis=1)
    labels_true = np.argmax(Z_true, axis=1)
  
    n = len(labels_hat)  

    A_true = np.equal.outer(labels_true, labels_true)   # 当 i, j 属于同一簇时为 True
    A_hat  = np.equal.outer(labels_hat, labels_hat)      # 当 i, j 属于同一簇时为 True


    mismatch_indicator = ((A_true.astype(int) + A_hat.astype(int)) == 1)

    error_count = np.sum(np.triu(mismatch_indicator, k=1))
    
    clustering_error = (2 / (n * (n - 1))) * error_count

    return 1 - clustering_error
