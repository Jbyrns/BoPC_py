# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:26:16 2018

@author: jeffr
"""
import numpy as np

class set_R_3D:
    

    @classmethod
    def set_R_3D(obj,event):
        """ Définition de Nbi et handles comme variables globales dans chaque fonction et sous-fonction
        handles : identifiants des objets graphiques (vecteur) """
        
        global handles R
        
        """ valeur de R """
        
        R = handles.get(32);
        R = R.astype(np.float);
       