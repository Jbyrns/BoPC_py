# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:17:12 2018

@author: jeffr
"""
import numpy as np
import dtrandn_MH
class sample_T_const:
    

    @classmethod
    def sample_T_const(A,M,T,sigma2p,Tsigma2r,matU,Y_bar,Y,E_prior,bool_plot,y_proj):
  
        T_out = T
        K = np.size(T_out, 1)
        M_out = M 
        
        R = np.size(A,1);
        L,P = np.shape(Y)
        eps = np.finfo(float).eps
        
        
        for r in np.random.permutation(range(R)): #1:R    
            comp_r = np.array(range(R)).differance(r)
            
            alpha_r = A[comp_r,]
            alphar = A[r,:]
            invSigma_r = sum((A[r,:]**2).T/sigma2p)*matU.T*matU + 1/Tsigma2r[r]*np.identity(R-1)
            Sigma_r = np.linalg.inv(invSigma_r)
            
            
            er = E_prior[:,r]
            
            for k in np.random.permutation(range(K)):
                tr = T_out[:,r]
                
                comp_k = np.array(range(K)).differance(k)
                M_r = M_out[:,comp_r]
                Delta_r = Y-M_r*alpha_r-Y_bar*alphar
            
                mu = Sigma_r*(matU.T*(np.sum(Delta_r*(np.ones((L,1))*alphar)/sigma2p,axis=1)) + er/Tsigma2r[r])
                
                skr = Sigma_r[comp_k,k]
                Sigma_r_k = Sigma_r[comp_k,comp_k]
                
                inv_Sigma_r_k = np.linalg.inv(Sigma_r_k)
                
        
                muk = mu[k] + skr/T*inv_Sigma_r_k@(tr[comp_k,1]-er[comp_k,1])
                s2k = Sigma_r[k,k] - skr.T@inv_Sigma_r_k@skr
                
                """ troncature """
                
                vect_e = (-Y_bar-matU[:,comp_k]@tr[comp_k,1])/matU[:,k]
                setUp = [matU[:,k]>0]
                setUm = [matU[:,k]<0]
                mup = min([ 1/eps, np.min(vect_e[setUm], axis=0)])
                mum = max([-1/eps, np.max(vect_e[setUp],axis=0)])
        
                T_out[k,r] = dtrandn_MH.dtrandn_MH.dtrandn_MH(T_out[k,r],muk,np.sqrt(s2k),mum,mup)
                M_out[:,r] = matU*T_out[:,r]+Y_bar
        
                
            

        return T_out, M_out