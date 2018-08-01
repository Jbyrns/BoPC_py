# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:35:52 2018

@author: jeffr
"""

import numpy as np
import randnt
class dtrandn_MH:
    

    @classmethod
    def dtrandn_MH(X,Mu,Sigma,Mum,Mup):




        Mu_new = Mu - Mum;
        Mup_new = Mup -Mum;
        
        if Mu<Mup:
            Z= randnt.randnt.randnt(Mu_new,Sigma,1);
        else:
            
            delta = Mu_new - Mup_new;
            Mu_new = -delta;
            Z= randnt.randnt.randnt(Mu_new,Sigma,1);
            Z = -(Z-Mup_new );
        

        Z = Z+Mum;
        cond = (Z<=Mup) and  (Z>=Mum);
        X = (Z*cond + X*( not cond));
        return X