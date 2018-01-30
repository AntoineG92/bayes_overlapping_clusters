# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import beta
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import bernoulli


class IOMM():
    def __init__(self, N, K, D, N_iter, Z, X, theta, alpha_prior, omega = 10, copy_rows = 4,burning_period=3):
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
        self.K_hat=self.K
        self.Z_hat=np.copy(self.Z)
    
    def compute_norm_lh(self, Z, N, K):
        norm_lh = np.zeros(K)
        for k in range(K):
            for i in range(N):
                norm_lh[k] += self.likelihood_ber(Z,i,k)
            print("norm_lh[",k,"] = ",norm_lh[k])
        result = np.median(norm_lh)
        print("norm_lh =", result)
        return result
    
    def learning(self,random_walk):
        theta_accept=[]
        Z_hat_list=[]
        theta_hat=np.copy(self.theta)
        self.Z_hat=np.copy(self.Z)
        U = np.zeros([self.N,self.N])
        for j in range(self.N_iter):
            #initialize Z_temp and P_Z_temp at each iteration
            self.Z_temp = np.zeros([self.N,self.K_hat])
            self.P_Z = np.zeros([self.N,self.K_hat])
            print("iteration n°",j)
            #during burning period we do not update Z
            if j>self.burning_period:
                self.Z_temp, self.P_Z, self.Z_hat = self.update_clusters()
                U=U+np.dot(self.Z_hat,self.Z_hat.T)
                Z_hat_list.append(self.Z_hat)
                #create new extended matrix of theta
                theta_hat=np.zeros([self.Z_hat.shape[1],self.D])
                #fill theta_hat with elements of theta for subset K*D
                theta_hat[:self.K_hat,:self.D]=self.theta
                #fill the new rows with the prior beta()
                if self.Z_hat.shape[1] > self.K_hat:
                    for k_plus in range(self.K_hat,self.Z_hat.shape[1]):
                        print("k+",k_plus)
                        theta_hat[k_plus,:self.D]=beta.rvs(self.alpha_prior/self.K_hat,1,size=self.D)
                self.K_hat=self.Z_hat.shape[1] #new K_hat

            if random_walk==True:
                theta_new,accept_ratio = self.resample_theta_rw(theta_hat)
                self.theta = theta_new
                print("the acceptance rate was:",accept_ratio)
            else:
                theta_new,accept_ratio = self.resample_theta(theta_hat)
                self.theta = theta_new
                #print("the acceptance rate was:",accept_ratio)
            print(self.theta)
            theta_to_append = {}
            theta_to_append= np.copy(theta_new)
            theta_accept.append(theta_to_append)
        U = U / (self.N_iter-self.burning_period)
        
        return self.Z_hat,theta_accept,U,Z_hat_list
    
    def update_clusters(self):
        Z = np.copy(self.Z)
        P_Z = self.P_Z
        Z_hat_new=np.copy(self.Z_hat) #matrix Z_hat that will be updated with existing clusters before drawing new ones
        N_prop_cluster=[]
        for i in self.ind_test:
            print("i =",i)
            P_Z[i,:] = self.update_p_z_i(i, P_Z)
            Z_hat_new[i,:] = self.propose_new_clusters(i, P_Z)
            #Propose adding new clusters wrt Poisson(alpha/N)
            N_prop_cluster.append(poisson.rvs(self.alpha_prior/self.N))
            print("new clusters proposed for i:",N_prop_cluster)
        #We now have the new cluster proposal for all observations. We can create the new extended matrix Z_hat
        #create the bigger matrix from scratch. If max number of new clusters is below K_hat, do not extend the size
        self.Z_hat=np.zeros([self.N,self.K+max(self.K_hat-self.K,np.max(N_prop_cluster))])
        #fill the rows up to the current matrice size of Z
        self.Z_hat[:,:self.K_hat]=np.copy(Z_hat_new[:,:self.K_hat])
        #fill the remaining rows with ones according to the Poisson(alpha/N) draw
        ind=0
        #if we draw new clusters beyond the size of K_hat, fill these new cluster with ones
        if np.max(N_prop_cluster) > self.K_hat-self.K:
            for i in self.ind_test:
                print("Number of proposed clusters:",N_prop_cluster[ind])
                if N_prop_cluster[ind] > self.K_hat-self.K:
                    self.Z_hat[i,self.K_hat:self.K+N_prop_cluster[ind]]=np.ones(N_prop_cluster[ind]-(self.K_hat-self.K))
                ind=ind+1            
        return Z, P_Z, self.Z_hat
    
    def update_p_z_i(self, i, P_Z):
        print("___________1.compute probability of observation i taking category k_________")
        for k in range(self.K_hat):
            m_without_i_k = self.m_without_i_k(i,k)
            if m_without_i_k > 0 and self.Z_hat[i,k] == 0:#we care only about categories that are not yet considered for movie i
                print("k=",k)
                Z_cond = np.copy(self.Z_hat)
                Z_cond[i,k]=1
                P_Z_1=(m_without_i_k/self.N) * self.likelihood_ber(Z_cond,i,k) #/ self.norm_lh
                Z_cond[i,k]=0
                P_Z_0=((self.N-m_without_i_k)/self.N) * self.likelihood_ber(Z_cond,i,k)
                P_Z[i,k]=P_Z_1 / (P_Z_1 + P_Z_0)
                
                print("proba Z=1:",P_Z[i,k])
       
        return P_Z[i,:]    
        
        
    def m_without_i_k(self, i, k):
        result = 0
        for j in range(self.N):
            if j != i:
                result += self.Z_hat[j,k]
        
        return result
    
    def likelihood_ber(self, Z, i, k):
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
    
    def propose_new_clusters(self, i, P_Z):
        print("_________2.propose adding new clusters________")
        Z=np.copy(self.Z_hat)
        for k in range(self.K_hat):
            if Z[i,k]==0 and np.random.uniform(0,1)<P_Z[i,k]:
                print('accepted for k =', k)
                Z[i,k]=1
        return Z[i,:]
        
    def resample_theta(self,theta_hat):
        accept_rate=0
        theta = np.copy(theta_hat)
        a = self.alpha_prior / self.K_hat
        std_prop=0.1
        print("_______3.resample theta|Z,X using MHA_______")
        for d in range(self.D):
            #extract current theta_d at index k
            theta_current = theta[:,d]
            #if theta is too small or too close to one, redraw another theta so that theta_prop does not collapse
            for k in range(self.K_hat):
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
         
            for k in range(self.K_hat):
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
        accept_rate=accept_rate/(self.K_hat*self.D)
        return (theta,accept_rate)
    

    def resample_theta_rw(self,theta_hat):
        accept_rate=0
        theta = np.copy(theta_hat)
        a = self.alpha_prior / self.K_hat
        std_prop=0.1 #standard deviation of truncated normal RW proposal
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
            #random walk proposal, gaussian truncated to interval (0,1)
            theta_prop=truncnorm.rvs(a=(0-theta_current)/std_prop,b=(1-theta_current)/std_prop,
                                     loc=theta_current,scale=std_prop,size=self.K_hat)
            
            #joint prior BETA(alpha/K,1) density over current and proposed parameters
            prior_theta_current = beta.pdf(theta_current, a, 1)
            prior_theta_prop = beta.pdf(theta_prop, a, 1)
            
            #likelihood densities
            lh_theta_current = self.likelihood_ber_d(theta_current, d)
            lh_theta_prop = self.likelihood_ber_d(theta_prop, d)
           
            for k in range(self.K_hat):
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
        accept_rate=accept_rate/(self.K_hat*self.D)    
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
        lh=np.zeros(self.K_hat)
        log_theta_ratio = np.log(theta_vect/(1-theta_vect))
        
        temp = 0
        for i in range(self.N):
            temp += self.Z_hat[i,:] * self.X[i,d] * log_theta_ratio
        lh = np.exp(temp)
            
        return lh                   