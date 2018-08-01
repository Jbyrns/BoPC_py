# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:46:37 2018

@author: jeffr
"""

import numpy as np
import scipy.special as scispec
class trandn:
    

    @classmethod
    def trandn(Mu,Sigma):



        Mu    =  Mu.reshape((1, np.product(np.shape(Mu))))
        Sigma =  Sigma.reshape((1, np.product(np.shape(Sigma))))
        eps = np.finfo(float).eps
        T = len(Mu)
        
        U = np.random.random((T,1))
        V = scispec.erf(- Mu/(sqrt(2)*max(Sigma,eps)));
        X = Mu + sqrt(2*Sigma**2) * scispec.erfinv(-(((1-V)*U + V)==1)*eps+(1-V)*U + V);
        X = max(X,eps);
        return X