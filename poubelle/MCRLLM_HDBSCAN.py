# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:22:58 2018

@author: jeffr
"""





""" NOTE: Pour la modification du script aller à la ligne 183 et modifier selon 
l'emplacement de votre fichier txt (matrice) et  le nombre de composantes (nb_c) 
ainsi que le nombre d'itérations"""





import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import dm3_lib as dm3
from functools import partial
from tqdm import tqdm
import hdbscan
import time


import matplotlib


class HyperspectralSegmentationMCR_LLM:

       
    
    @classmethod
    def mcr_llm3d(cls, xi, nb_c, nb_iter =25):

        m,n,o = np.shape(xi)
        c_f = np.zeros((o*n, nb_c))
        x_f = np.zeros((o*n, m))
        for i in range(o):
            #x=xi[:,i,:].T
            #[s, x_sum, x_norm, x_raw, ci] = cls.initialisation(x, nb_c)
            #c_f[i*o:(i+1)*o,:np.size(ci,axis=1)] = ci
            x_f[i*n:(i+1)*n,:] = xi[:,:,i].T

        #x_norm, x_sum = cls.initialisation_2d(x_f)
        start = time.time()
        s, x_sum, x_norm, x_raw, c_f = cls.initialisation(x_f, nb_c)
        end = time.time()
        tottime = (end-start)//60
        print(tottime)

        for nb_iter in tqdm(range(nb_iter)):
            # Concentrations (with Poisson likelihood maximization)
            
            c = cls.C_plm(x_f, s, x_norm, x_sum, nb_c, c_f)
            s = cls.s_plm(x_norm, c)
            c_f = c


        return c,s







    @classmethod
    def initialisation_2d(cls, x):

        x_sum = np.asarray([np.sum(x, axis=1)]).T
        x_norm = x / x_sum
        x_norm = np.nan_to_num(x_norm)
        x_raw = x  # note that la
        return x_norm, x_sum


    @classmethod
    def initialisation(cls, x, nb_c):

        x_sum = np.asarray([np.sum(x, axis=1)]).T
        x_norm = x / x_sum
        x_norm = np.nan_to_num(x_norm)
        x_raw = x  # note that lambda is a function in python, lada replaces lambda as not to overwrite it. Also lada is a shitty russian car manufacturer, look it up!



        for mcs in range(2,10):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
            cluster_labels = clusterer.fit_predict(x_norm)
            
            number_of_components = len(np.unique(cluster_labels))
            mcs+=1
        
            if number_of_components == nb_c+1: #The addition of a component is necessary since this method clusters all noise in its -1 cluster label
                break  
          
        s = np.zeros(( np.size(x_norm, axis = 1),nb_c))
        
        for j in range(nb_c):
            filter1 = cluster_labels==j
            spec = np.mean(x_norm[filter1,:],axis =0).T
            s[:,j-1] = spec.T
               
        
        s = s.T
        
                       
        
        c_pred = x_norm @ s.T @ np.linalg.inv(s @ s.T)
        return s, x_sum, x_norm, x_raw, c_pred
    

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
    
dm3f = dm3.DM3("EELS Spectrum Image (dark ref corrected).dm3")
x = dm3f.imagedata
nb_c =3  # nombre de composantes
nb_iter = 25 # nombre d'itérations

[c, s] = HyperspectralSegmentationMCR_LLM.mcr_llm3d(x,nb_c,nb_iter) # appel de l'algorithme

matplotlib.pyplot.plot(c)




