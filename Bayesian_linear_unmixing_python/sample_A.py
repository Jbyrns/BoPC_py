# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:42:39 2018

@author: jeffr
"""

import numpy as np
import dtrandnmult

class sample_A:
    

    @classmethod
    def sample_A(X,S,A,R,P,sigma2e):



        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
         This function allows to sample the abundances 
               according to its posterior f(A|...)
         USAGE
               A = sample_abundances(X,S,A,R,P,L,sigma2e,mu0,rhoa,psia)
         
         INPUT
               X,S,A : matrices of mixture, sources and mixing coefficients 
                R,P,L  : number of sources, observations and samples
               sigma2e : the current state of the sigma2 parameter
               rho,psi  : hyperprior parameters 
         
         OUTPUT
               A     : the new state of the A parameter
         
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        y = X;
        
        ordd = np.random.permutation(range(R)) 
        jr = ordd[-1]
        comp_jr = ordd[1:-2]
        alpha = A[:,comp_jr]
        
        """ useful quantities """
        u = np.ones(R-1).T
        M_R = S[jr,:].T
        M_Ru = M_R@u.T
        M = S[comp_jr,:].T
        T = (M-M_Ru).T@(M-M_Ru)
        
        for p in range(P):
            Sigma = np.linalg.inv(T/sigma2e);
            Mu    = Sigma@((1/sigma2e)@(M-M_Ru).T@(y[p,:].T-M_R))
            alpha[p,:] = dtrandnmult.dtrandnmult.dtrandnmult(alpha[p,:],Mu,Sigma)

        A[:,ordd[:-1]] = alpha
        A[:,ordd[-1]] = np.max(1-np.sum(alpha,axis=1),0);
        return A