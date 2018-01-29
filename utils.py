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

def get_mat_random(mat_x, list_rows):
    mat_y = np.zeros((mat_x.shape[0], mat_x.shape[1]))
    for i in range(mat_x.shape[0]):
        mat_y[i,:] = mat_x[list_rows[i],:]
    return mat_y

def clear_users(data):    
    nbr_clear = 0
    list_col = []
    for i in range(data.shape[1]):
        temp_ratings = data[:,i]
        nbr_ratings = sum(temp_ratings)    
        if nbr_ratings > 20: 
            list_col.append(i)
            #data = np.delete(data, i, 1)
        else:
            nbr_clear += 1
    if len(list_col)>0:
        data_cleaned = data[:,list_col]
    else:
        data_cleaned = data
    return data_cleaned, nbr_clear

def clear_movies(data):
    nbr_clear = 0
    list_row = []
    for i in range(data.shape[0]):
        temp_ratings = data[i,:]
        nbr_ratings = sum(temp_ratings)    
        if nbr_ratings > 10: 
            list_row.append(i)
            #data = np.delete(data, i, 0)
        else:
            nbr_clear += 1
    if len(list_row)>0:
        data_cleaned = data[list_row,:]
    else:
        data_cleaned = data
    return data_cleaned, nbr_clear