# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as npr
from scipy.stats import beta

def random_z(N, K):
    #randomize Z matrix
    Z=np.zeros([N,K])
    for i in range(N):
        ind1=int(npr.uniform(0,K))
        ind2=int(npr.uniform(0,K))
        ind3=int(npr.uniform(0,K))
        Z[i,ind1]=1
        if(npr.uniform(0,1)<0.5): #assume 50% chance of taking a second category
            Z[i,ind2]=1
        if(npr.uniform(0,1)<0.1): #assume 10% chance of taking a third category
            Z[i,ind3]=1
    return Z


def random_x(N, D):
    #randomize X matrix
    X = np.zeros([N,D])
    for i in range(N):
        ind_dim=npr.uniform(0,1,D)
        for j in range(D):  #fill each row with randomly assign ones, let 1/3 proba of a given dimension to be = 1
            if ind_dim[j]<0.3:
                X[i,j]=1
        if np.sum(X[i,:])==0:  #at least one dimension equal one to avoid empty observations
            X[i,int(npr.uniform(0,D))]=1
    return X


def random_theta(N, K, D, alpha_prior):
    #randomize theta based on beta(alpha/K,1) prior
    theta=np.zeros([K,D])
    for k in range(K):
        theta[k,:]=beta.rvs(alpha_prior/K,1,size=D)
    return theta