#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 2
# Module : MDA implementation
#=======================================================

import numpy as np

# calculating the ML estimate
def calc_ML_estimates(x,y):
    mu = []
    sigma = []
    sigma_inv = []
    sigma_det = []
    threshold = 0.0000001
    for i in set(y):
        idx = np.where(y == i)
        mu_i = np.mean(x[idx], axis=0)
        sigma_i = np.cov(x[idx], rowvar=False)*(1/2)
        mu.append(mu_i)
        if np.linalg.det(sigma_i) < 0.00001:
            w, v = np.linalg.eig(sigma_i)
            sigma_det.append(np.product(np.real(w[w > threshold])))
            sigma_i = sigma_i + 0.0001*np.eye(len(mu_i))
            sigma.append(sigma_i)
            sigma_inv.append(np.linalg.inv(sigma_i))
        else:
            sigma.append(sigma_i)
            sigma_det.append(np.linalg.det(sigma_i))
            sigma_inv.append(np.linalg.inv(sigma_i))

    return mu, sigma, sigma_det, sigma_inv

def MDA(x,y):
    n_classes = len(set(y))
    prior = 1/n_classes
    features = x.shape[1]
    # calculate mu and sigma for each class
    mu,sigma,_,_ = calc_ML_estimates(x,y)
    mu = np.array(mu)
    # calculate the anchor mean
    anchor_mean = np.sum(prior*mu,axis = 0).reshape((1,features))
    sigma_b = np.zeros((features,features))
    sigma_w = np.zeros((features,features))
    # calculate sigma_b and sigma_W
    for i in range(n_classes):
        sigma_b += prior*np.matmul((mu[i]-anchor_mean).T, mu[i]- anchor_mean)
        sigma_w += prior*sigma[i]
    
    sigma_w += 0.0001*np.eye(features)
    eig_val, eig_vec = np.linalg.eig(np.matmul(np.linalg.inv(sigma_w), sigma_b))
    idx = np.argsort(np.real(eig_val))[::-1]
    sorted_eigenvectors = eig_vec[:,idx]
    non_zero = np.count_nonzero(np.real(eig_val)>1e-10)
    print('The non zero elements are ',non_zero)
    A = sorted_eigenvectors[:,0:non_zero]
    theta = (1/features)*A
    z = (np.matmul(theta.T, x.T)).T
    # x_mda = np.matmul(z, theta.T)

    return np.real(z), theta


