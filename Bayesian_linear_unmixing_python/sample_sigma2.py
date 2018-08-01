# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:32:45 2018

@author: jeffr
"""
import numpy as np

class sample_sigma:
    

    @classmethod
    def sample_sigma2(A_est,M_est,Y):  
        
        L,P = np.shape(Y)
        
        tmp = Y - M_est*A_est
        
        coeff1 = P*L/2
        coeff2 = sum((sum(tmp**2,1)/2)).T

        
        
        Tsigma2p = 1/np.stats.gamma.rvs(coeff1,1/coeff2)
        return  Tsigma2p