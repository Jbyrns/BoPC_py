# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:07:16 2018

@author: jeffr
"""

import numpy as np
import trandn
import dtrandn_MH
import scipy.linalg.inv as inv
class dtrandnmult:
    

    @classmethod
    def dtrandnmult(S,Mu,Re):



        """ Missing data sampling """
        S =  S.reshape((1, np.product(np.shape(S))))
        Mu =  Mu.reshape((1, np.product(np.shape(Mu))))
        Sigma_mat  = {}
        Sigma_vect = {}

        R = len(S)
        if R==1:
            S = trandn.trandn.trandn(Mu,np.sqrt(Re))
        else:
            
            for r in range(R):
                Rm = Re
                Rm = np.delete(Rm,r,0)
                Rv = Rm[:,r]
                Rm = np.delete(Rm,r,1)
                Sigma_mat[r]  = inv(Rm)
                Sigma_vect[r] = Rv

            Moy_Sv = np.array(range(R))
            Var_Sv = np.array(range(R))
            Std_Sv = np.array(range(R))
            for iter in range(5):
                for k in np.random.permutation(range(R)):
                    Sk = S
                    Sk = np.delete(Sk,k,0)
                    Muk = Mu
                    Muk = np.delete(Muk,k,0)
                    Moy_Sv[k]  = Mu[k] + Sigma_vect[k].T@Sigma_mat[k]@(Sk-Muk);
                    Var_Sv[k]   = Re[k,k] - Sigma_vect[k].T@Sigma_mat[k]@Sigma_vect[k];
                    Std_Sv[k]   = np.sqrt(abs(Var_Sv[k]));
                    S[k] = dtrandn_MH.dtrandn_MH.dtrandn_MH(S[k],Moy_Sv[k],Std_Sv[k],0,[1-sum(S)+S[k]]);
        
        return S