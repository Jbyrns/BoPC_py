# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:22:58 2018

@author: jeffr
"""





""" NOTE: Pour la modification du script aller à la ligne 78 et modifier selon 
l'emplacement de votre fichier txt (matrice) et  le nombre de composantes (nb_c) 
ainsi que le nombre d'itérations"""





import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import Cyplmtot 
import matplotlib


class HyperspectralSegmentationMCR_LLM:

       
    
    @classmethod
    def mcr_llm(cls, x, nb_c, nb_iter =25):
    

        
        [s, x_sum, xnorm, lada] = cls.initialisation(x, nb_c)

        for nb_iter in tqdm(range(nb_iter)):
            # Concentrations (with Poisson likelihood maximization)
            #c,s = Cyplmtotnew.C_plm(x, s, lada,x_sum, nb_c)
            c,s = Cyplmtot.C_plm(x, s, lada, x_sum, nb_c)
            



        return c, s



    @classmethod
    def initialisation(cls, x, nb_c):

        x_sum = np.asarray([np.sum(x, axis=1)]).T
        xnorm = x / x_sum
        xnorm = np.nan_to_num(xnorm)
        lada = x  # note that lambda is a function in python, lada replaces lambda as not to overwrite it. Also lada is a shitty russian car manufacturer, look it up!


        c = KMeans(n_clusters=nb_c).fit(xnorm).labels_
        s = np.zeros(( np.size(xnorm, axis = 1),nb_c))
        for j in range(0,nb_c):
            filter1 = c==j
            spec = np.mean(xnorm[filter1,:],axis =0).T
            s[:,j-1] = spec.T
            
           

        s = s.T
        
            
        


        return s, x_sum, xnorm, lada

   

    
x = np.loadtxt("data_2.txt", comments="#", delimiter=",", unpack=False) # Importation de la matrice
x = x.T
nb_c =7  # nombre de composantes
nb_iter = 25
 # nombre d'itérations

[c, s] = HyperspectralSegmentationMCR_LLM.mcr_llm(x,nb_c,nb_iter) # appel de l'algorithme


matplotlib.pyplot.plot(c)


