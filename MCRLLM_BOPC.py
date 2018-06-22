# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:22:58 2018

@author: jeffr
"""





""" Modify line 186 to select datafile to analyse, line 185 to modify the correct 
direction of the file (i.e. the observations must be attributed to the rows and 
the pixels to the columns),line 186 to modify the number of components and line 
103 to modify the number of iterations made by the algorithm (passing between c and s).
For any and all information please contact me at jeffrey.byrns@usherbrooke.ca""" 




import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import scipy.stats as stats

from functools import partial
from tqdm import tqdm


import matplotlib


class HyperspectralSegmentationMCR_LLM:

       
    
    @classmethod
    def mcr_llm(cls, x, nb_c, nb_iter =25):
    

        
        [s, x_sum, x_norm, x_raw] = cls.initialisation(x, nb_c)
        c_pred = x_norm @ s.T @ np.linalg.inv(s @ s.T)
        #c_pred[c_pred<0]=0
        #c_pred = c_pred**1.2
        #c_pred = (c_pred.T/np.sum(c_pred, axis =1)).T

        for nb_iter in tqdm(range(nb_iter)):
            # Concentrations (with Poisson likelihood maximization)
            
            c = cls.C_plm(x, s, x_norm, x_sum, nb_c, c_pred)
            s = cls.s_plm(x, c)
            



        return c, s



    @classmethod
    def initialisation(cls, x, nb_c):

        x_sum = np.asarray([np.sum(x, axis=1)]).T
        x_norm = x / x_sum
        x_norm = np.nan_to_num(x_norm)
        x_raw = x  

        c = KMeans(n_clusters=nb_c).fit(x_norm).labels_
        s = np.zeros(( np.size(x_norm, axis = 1),nb_c))
        for j in range(0,nb_c):
            filter1 = c==j
            spec = np.mean(x_norm[filter1,:],axis =0).T
            s[:,j-1] = spec.T
            
           

        s = s.T

        return s, x_sum, x_norm, x_raw
    

    @classmethod
    def s_plm(cls,x,c):
        S2 = np.linalg.inv(c.T@c)@c.T@x# Multilinear regression
        S2[S2 <= np.spacing(1)] = np.spacing(1) # avoid 0 values
        s = S2/ (np.sum(S2, axis =1) * np.ones((np.size(S2, axis = 1),1))).T# Normalization
        return s

    
    
    @classmethod
    def C_plm(cls,x, s, xnorm,x_sum,nb_c, c_pred):
          #initialize C

        [n,m] = np.shape(x)
        c_new = np.zeros((n,nb_c))
        
        c_iter_mod = c_new
        #Paralled FOR loops
        

        #with ThreadPoolExecutor(max_workers=cpus) as executor:
        #for ids in range(0,n):
            
        for ids in range(0,n):
  
                
                #a= executor.submit(partial(cls.pyPLM,nb_c,s),xnorm[ids,:])
                #C[ids,:] = a.result() 
                
                if ids<5:
                    c_new[ids,:], c_iter_mod[ids,:] = cls.pyPLM(nb_c, s, xnorm[ids,:], c_pred[ids,:], boolin=False)
                else:
                    c_new[ids,:], c_iter_mod[ids,:] = cls.pyPLM(nb_c, s, xnorm[ids,:], c_pred[ids-5:ids,:], boolin=True) 
        
        # avoid errors (this part should not be necessary)
        c_new[np.isnan(c_new)] = 1/nb_c
        c_new[np.isinf(c_new)] = 1/nb_c
        c_new[c_new<0] = 0
        c_sum1 = np.array([np.sum(c_new,axis=1)])
        c_sum =c_sum1.T@np.ones((1,np.size(c_new,axis =1)))
        c_new = c_new/c_sum

        return c_new
    
    
    @classmethod
    def pyPLM(cls, nb_c, s, x_norm, c,boolin=False):
        c_old = c

        yObs = x_norm
        yObs = np.array([yObs])

        # sum of every value is equal to 1
        def con_one(c):
            return 1-sum(c) 
        

        # all values are positive
        bnds = ((0.0, 1.0),) * nb_c

        cons = [{'type': 'eq', 'fun': con_one}]

        def regressLL(s, yObs, c_pred):
            
          
            yPred = c_pred @ s
            

            # Calculate the negative log-likelihood as the negative sum of the log of a normal
            # PDF where the observed values are normally distributed around the mean (yPred)
            # with a standard deviation of sd
            logLik = -np.sum(stats.norm.logpdf(yObs, loc=yPred))

            # Tell the function to return the NLL (this is what will be minimized)
            return (logLik)

        try:
            c_pred = x_norm @ s.T @ np.linalg.inv(s @ s.T)


            
            if boolin:
                cavc = np.average(c_old, axis = 0)*c_pred
                c_pred = cavc/sum(cavc)
            
        except np.linalg.LinAlgError:

            cavc = np.average(c_old, axis = 0)
            c_pred = cavc/sum(cavc)
            
            pass
        
        # Run the minimizer
        results = minimize(partial(regressLL, s, yObs), c_pred, method='SLSQP', bounds=bnds, constraints=cons, jac =False)
        results = results.x
        results = np.asarray(results)

        c_new = results.reshape(int(len(results) / nb_c), nb_c)
        return c_new,c_pred
    
    
x = np.loadtxt("data_2.txt", comments="#", delimiter=",", unpack=False) # Importation de la matrice
x = x.T
nb_c =7  # nombre de composantes
nb_iter = 50 # nombre d'itérations

[c, s] = HyperspectralSegmentationMCR_LLM.mcr_llm(x,nb_c,nb_iter) # appel de l'algorithme

matplotlib.pyplot.plot(c)




