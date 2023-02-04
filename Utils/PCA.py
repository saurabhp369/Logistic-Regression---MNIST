#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 2
# Module : PCA implementation
#=======================================================

import numpy as np

def PCA(x):
    N = x.shape[0]
    std_x = (x - np.mean(x, axis = 0))
    sigma = (N-1)/ N * np.cov(std_x, rowvar = False)
    print(sigma.shape)
    # Eigendecomposition of covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(sigma) 
    # Find Required no.of components 
    keep_var = 0.95 # Minimum fraction of Variance to keep
    req_var = keep_var*sum(eig_vals)
    req_dim = 0
    variance = 0
    for i in range(len(eig_vals)):
      variance += np.abs(eig_vals[i])
      if variance >= req_var:
          req_dim = i + 1
          m = req_dim
          break
    print('Required Dimension for PCA: ', m)

    idx = np.argsort(np.real(eig_vals))[::-1]
    eig_vecs = eig_vecs[:,idx]
    eig_vecs_new = eig_vecs[:, :m]
    X_proj = np.real(np.matmul(x, eig_vecs_new))
    return X_proj, eig_vecs_new
