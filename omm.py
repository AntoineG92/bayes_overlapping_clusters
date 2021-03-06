# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import beta
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import bernoulli

class IOMM():
    def __init__(self, N, K, D, N_iter, Z, X, theta, alpha_prior, omega = 10, copy_rows=3,burning_period=3):
        self.N = N
        self.K = K
        self.D = D
        self.N_iter = N_iter
        self.P_Z = np.zeros([N,K])
        self.X = X
        self.theta = theta
        self.burning_period=burning_period        
        self.alpha_prior = alpha_prior
        self.omega = omega
        self.copy_rows = copy_rows
        ind_train=np.random.randint(0,self.N,self.copy_rows)
        self.ind_train=ind_train
        ind_test=np.delete(np.arange(self.N),self.ind_train)
        self.ind_test=ind_test
        self.Z = np.zeros([N,K])
        self.Z[ind_train,:] = Z[ind_train,:]
        
        self.Z_temp = np.zeros([self.N,self.K])
    

    def learning(self,random_walk):
        print("ind test",self.ind_test)
        theta_accept=[]
        Z_mean=np.zeros([self.N,self.K])
        U = np.zeros([self.N,self.N])
        for j in range(self.N_iter):
            #initialize Z_temp and P_Z_temp at each iteration
            self.Z_temp = np.zeros([self.N,self.K])
            self.P_Z = np.zeros([self.N,self.K])
            print("iteration n°",j)
            #during burning period we do not update Z
            if j>self.burning_period:
                self.Z_temp, self.P_Z = self.update_clusters()
                Z_mean=self.Z_temp+Z_mean
                U=U+np.dot(self.Z_temp,self.Z_temp.T)
            if random_walk==True:
                theta_new,accept_ratio = self.resample_theta_rw()
                self.theta = theta_new
                print("the acceptance rate was:",accept_ratio)
            else:
                theta_new,accept_ratio = self.resample_theta()
                self.theta = theta_new
                print("the acceptance rate was:",accept_ratio)
            print(self.theta)
            theta_to_append = {}
            theta_to_append= np.copy(theta_new)
            theta_accept.append(theta_to_append)
        Z_mean=Z_mean/(self.N_iter-self.burning_period)
        U = U / (self.N_iter-self.burning_period)
        
        return self.Z_temp,theta_accept,Z_mean,U
    
    def update_clusters(self):
        Z = np.copy(self.Z)
        P_Z = self.P_Z
        
        for i in self.ind_test:
            print("i =",i)
            P_Z[i,:] = self.update_p_z_i(i, P_Z, Z)
            Z[i,:] = self.propose_new_clusters(i, P_Z, Z)
            
        return Z, P_Z
    
    def update_p_z_i(self, i, P_Z, Z):
        print("___________1.compute probability of observation i taking category k_________")
        
        for k in range(self.K):
            m_without_i_k = self.m_without_i_k(Z,i,k)
            if m_without_i_k > 0 and Z[i,k] == 0:  #we care only about categories that are not yet considered for movie i
                print("k=",k)
                Z_cond = np.copy(Z)
                Z_cond[i,k]=1
                P_Z_1=(m_without_i_k/self.N) * self.likelihood_ber(Z_cond,i,k) #/ self.norm_lh
                Z_cond[i,k]=0
                P_Z_0=((self.N-m_without_i_k)/self.N) * self.likelihood_ber(Z_cond,i,k)
                P_Z[i,k]=P_Z_1 / (P_Z_1 + P_Z_0)
                print("proba Z=1:",P_Z[i,k])
       
        return P_Z[i,:]    
        
        
    def m_without_i_k(self, Z, i, k):
        result = 0
        for j in range(self.N):
            if j != i:
                result += Z[j,k]
        
        return result
    
    def likelihood_ber(self, Z, i,k):
    #LIKELIHOOD density of observation i, k fixed
        result=1
        num=1
        den1=1
        for d in range(self.D):
            for k in range(self.K):  #compute theta_d equation (7)
                num=num*self.theta[k,d]**Z[i,k]
                den1=den1*(1-self.theta[k,d])**Z[i,k]
                theta_d=num/(den1+num)
            result=result*bernoulli.pmf(k=self.X[i,d],p=theta_d) #compute likelihood
        return result
    
    def propose_new_clusters(self, i, P_Z, Z):
        print("_________2.propose adding new clusters________")
        for k in range(self.K):
            if Z[i,k]==0 and np.random.uniform(0,1)<P_Z[i,k]:
                print('accepted for k =', k)
                Z[i,k]=1
        return Z[i,:]
        
    def resample_theta(self):
        accept_rate=0
        theta = np.copy(self.theta)
        a = self.alpha_prior / self.K
        std_prop=0.1
        print("_______3.resample theta|Z,X using MHA_______")
        for d in range(self.D):
            #extract current theta_d at index k
            theta_current = theta[:,d]
            #if theta is too small or too close to one, redraw another theta so that theta_prop does not collapse
            for k in range(self.K):
                while (theta_current[k] < 10**(-2) or theta_current[k] > 0.95):
                    print("redraw theta",k)
                    theta_current[k]=beta.rvs(a,1)
            
            #draw a proposal parameter centered around its current value
            theta_prop = self.proposal_beta(theta_current)
            
            #joint prior BETA(alpha/K,1) density over current and proposed parameters
            prior_theta_current = beta.pdf(theta_current, a, 1)
            prior_theta_prop = beta.pdf(theta_prop, a, 1)
            
            #likelihood densities
            lh_theta_current = self.likelihood_ber_d(theta_current, d)
            lh_theta_prop = self.likelihood_ber_d(theta_prop, d)
            
            for k in range(self.K):
                #transition probabilities theta|theta_prop and theta_prop|theta
                trans_theta_current = self.trans_proba_beta(theta_current, theta_prop, k)
                trans_theta_prop = self.trans_proba_beta(theta_prop, theta_current, k)
                #accept/reject probability
                numerator = np.dot(lh_theta_prop,prior_theta_prop) * trans_theta_current
                denominator = np.dot(lh_theta_current,prior_theta_current) * trans_theta_prop
                accept_proba= numerator / denominator
                
                if np.random.uniform(0,1)< min(accept_proba,1):
                    theta[k,d]=theta_prop[k]
                    print("accept")
                    accept_rate=accept_rate+1
        accept_rate=accept_rate/(self.K*self.D)
        return (theta,accept_rate)
    
    def resample_theta_rw(self):
        accept_rate=0
        theta = np.copy(self.theta)
        a = self.alpha_prior / self.K
        std_prop=0.1 #standard deviation of truncated normal RW proposal
        print("_______3.resample theta|Z,X using MHA_______")
        for d in range(self.D):
            #extract current theta_d at index k
            theta_current = theta[:,d]
            #if theta is too small or too close to one, redraw another theta so that theta_prop does not collapse
            for k in range(self.K):
                while (theta_current[k] < 10**(-3) or theta_current[k] > 0.95):
                    print("redraw theta",k)
                    theta_current[k]=beta.rvs(a,1)
            
            #draw a proposal parameter centered around its current value
            #random walk proposal, gaussian truncated to interval (0,1)
            theta_prop=truncnorm.rvs(a=(0-theta_current)/std_prop,b=(1-theta_current)/std_prop,
                                     loc=theta_current,scale=std_prop,size=self.K)
            
            #joint prior BETA(alpha/K,1) density over current and proposed parameters
            prior_theta_current = beta.pdf(theta_current, a, 1)
            prior_theta_prop = beta.pdf(theta_prop, a, 1)
            
            #likelihood densities
            lh_theta_current = self.likelihood_ber_d(theta_current, d)
            lh_theta_prop = self.likelihood_ber_d(theta_prop, d)
           
            for k in range(self.K):
                #transition probabilities theta|theta_prop and theta_prop|theta
                trans_theta_current = norm.cdf(theta_current[k]/theta_prop[k],loc=0,scale=1)
                trans_theta_prop = norm.cdf(theta_prop[k]/theta_current[k],loc=0,scale=1)
                #accept/reject probability
                numerator = np.dot(lh_theta_prop,prior_theta_prop) * trans_theta_current
                denominator = np.dot(lh_theta_current,prior_theta_current) * trans_theta_prop
                accept_proba= numerator / denominator
                
                if np.random.uniform(0,1)< min(accept_proba,1):
                    print("accept")
                    theta[k,d]=theta_prop[k]
                    accept_rate=accept_rate+1
        accept_rate=accept_rate/(self.K*self.D)    
        return (theta,accept_rate)
    
    def proposal_beta(self, theta_d):
        omega = self.omega
        return (beta.rvs(omega*theta_d,omega*(1-theta_d)))
    
    def trans_proba_beta(self, theta, theta_param, k):
    #transition probability
        omega = self.omega
        theta_param_value = theta_param[k]
        theta_value = theta[k]
        return (beta.pdf(theta_value,omega*theta_param_value,omega*(1-theta_param_value)))
    
    def likelihood_ber_d(self, theta_vect, d):
    #LIKELIHOOD OF K DIMENSIONAL ARRAY (for MHA algo)        
        lh=np.zeros(self.K)
        log_theta_ratio = np.log(theta_vect/(1-theta_vect))
        
        temp = 0
        for i in range(self.N):
            temp += self.Z_temp[i,:] * self.X[i,d] * log_theta_ratio
        lh = np.exp(temp)
            
        return lh                   