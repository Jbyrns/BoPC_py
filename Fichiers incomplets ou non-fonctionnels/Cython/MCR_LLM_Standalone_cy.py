# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:22:58 2018

@author: jeffr
"""





""" NOTE: Pour la modification du script aller à la ligne 298 et modifier selon 
l'emplacement de votre fichier txt (matrice) et  le nombre de composantes (nb_c) 
ainsi que le nombre d'itérations"""





import numpy as np
from sklearn.cluster import KMeans
from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing
import Cyplmtot
import matplotlib



class HyperspectralSegmentationMCR_LLM:

       
    
    @classmethod
    def mcr_llm(cls, x, nb_c, nb_iter =25):


        [s, x_sum, xnorm, lada] = cls.initialisation(x, nb_c)

        for nb_iter in tqdm(range(nb_iter)):
            # Concentrations (with Poisson likelihood maximization)
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

    @classmethod
    def s_plm(cls,x,c):

        S2 = np.linalg.inv(c.T@c)@c.T@x# Multilinear regression
        S2[S2 <= np.spacing(1)] = np.spacing(1) # avoid 0 values
        s = S2/ (np.sum(S2, axis =1) * np.ones((np.size(S2, axis = 1),1))).T# Normalization

        return s

    
    
    @classmethod
    def C_plm(cls,x, s, lada,x_sum,nb_c):
        
        A = np.array(np.append(-np.eye(nb_c),np.ones((1,nb_c)),axis = 0))
        b = np.array([np.append(np.zeros((nb_c,1)),1)]).T
        



         #initialize C

        [n,m] = np.shape(x)
        C = np.zeros((n,nb_c))

        #FOR ALL PIXELS
        #Paralled FOR loop

        p = Pool(4)
        #error_report = np.zeros((n,1), dtype =bool)

        
        
        C = p.map(Cyplm.plm, ((nb_c,x_sum[ids,0],s,lada[ids,:],x_sum[ids,0]*s,A,b) for ids in range(n)))
 
            #C[ids,:] = cls.pyPLM(lada[ids,:],nb_c, s)

        
        # avoid errors (this part should not be necessary)

        C[np.isnan(C)] = 1/nb_c
        C[np.isinf(C)] = 1/nb_c
        C[C<0] = 0
        C_sum1 = np.array([np.sum(C,axis=1)])
        C_sum =C_sum1.T@np.ones((1,np.size(C,axis =1)))
        C = C/C_sum

        
        return C
    
    
    
x = np.loadtxt("data_2.txt", comments="#", delimiter=",", unpack=False) # Importation de la matrice
x = x.T
nb_c =7  # nombre de composantes
nb_iter = 25
 # nombre d'itérations

[c, s] = HyperspectralSegmentationMCR_LLM.mcr_llm(x,nb_c,nb_iter) # appel de l'algorithme

matplotlib.pyplot.plot(c)





