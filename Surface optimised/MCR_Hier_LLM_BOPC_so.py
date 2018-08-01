# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:22:58 2018

@author: jeffr
"""



""" This algorithm performs MCR-Hier-LLM on the selected data set. 
To use, select a dataset in by overwriting the name of the dataset of line 291.
Please modify line 292 to modify the correct direction of the file 
(i.e. the observations must be attributed to the rows and the pixels to the columns),
then set the nuber of components to be used on line 293. For any and all information
please contact me at jeffrey.byrns@usherbrooke.ca"""



import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import scipy.stats as stats
from functools import partial
import dm3_lib as dm3




class HyperspectralSegmentationMCR_HIER_LLM:

       
    
    @classmethod
    def mcr_llm(cls, x, nb_c, nb_iter =25):
    

        
        [s, x_sum, x_norm, x_raw] = cls.initialisation(x, nb_c)
        c_pred = x_norm @ s.T @ np.linalg.inv(s @ s.T)
        #c_pred[c_pred<0]=0
        #c_pred = c_pred**1.2
        #c_pred = (c_pred.T/np.sum(c_pred, axis =1)).T

        for nb_iter in range(nb_iter):
            # Concentrations (with Poisson likelihood maximization)
            
            c = cls.C_plm(x, s, x_norm, x_sum, nb_c, c_pred)
            s = cls.s_plm(x, c)
            
            c_pred = c
            


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
    
    @classmethod
    def mcr_hier(cls, x, max_iter=25, max_level=10):

        obs = np.size(x,axis=0)
        min_pixels = max(50, obs*.01)

        # First level
        
        Extracted_conc = []  # list
        Extracted_pos = []  # list
        Extracted_Spectra = []  # array
        Extracted_end = []  # array
        
        # Generate position vector
        
        positions = np.array(range(np.size(x, axis=0)))
        
        #call mcr_llm
        [c, s] = cls.mcr_llm(x, 2, max_iter)

        # list initialisation
        
        all_spectra = [[0 for x in range(2+(2**max_level-1)*2) ] for y in range(max_level+1) ]  # list
        all_conc =[[0 for x in range(2+(2**max_level-1)*2) ] for y in range(max_level+1) ]  # list
        all_pos = [[0 for x in range(2+(2**max_level-1)*2) ] for y in range(max_level+1) ]  # list
        
        # Save data
        all_spectra[0][0] = s[0, :]
        all_spectra[0][1] = s[1, :]
        all_conc[0][0] = c[:, 0]
        all_conc[0][1] = c[:, 1]
        
        # Save filters
        
        all_pos[0][0] = positions[c[:, 0] > .5]
        all_pos[0][1] = positions[c[:, 0] <= .5]
        
        """ Next LEVELS """
        current_level = 1
        
        all_count = 1
        
        for level_nb in range(max_level):
            # for each division
            for level in range(2 ** (current_level)):
                # Get positions
        
                try:
                    cur_pos = all_pos[current_level-1][level]
                except not cur_pos == 0:
                    pass
            # Continue if it is ok
        
                if np.size(cur_pos) > 1:
            
                    if len(cur_pos) >= min_pixels:
                        #call mcr_llm
                        [c, s] = cls.mcr_llm(x[cur_pos,:], 2, max_iter)
            
                        # save data
                        all_spectra[current_level][2+(level-1) * 2] = s[0, :]
                        all_spectra[current_level][3 + (level-1) * 2] = s[1, :]
                        all_conc[current_level][2+(level-1) * 2] = c[:, 0]
                        all_conc[current_level][3 + (level-1) * 2] = c[:, 1]
            
                        # save filters
                        all_pos[current_level][2+(level-1) * 2] = cur_pos[c[:, 0] > .5]
                        all_pos[current_level][3 + (level-1) * 2] = cur_pos[c[:, 0] <= .5]
                    else:  # just reached the minimum pixels
            
                        # add concerned spectrum
                        Extracted_Spectra = np.append(Extracted_Spectra, all_spectra[current_level-1][level])
        
                        Extracted_end = np.append(Extracted_end, [current_level-1, level], axis=0)
            
                        CC = all_conc[current_level-1][level]
            
                        Extracted_conc.append(CC[CC > .5])
                        Extracted_pos.append(all_pos[current_level-1][level])
                        all_count = all_count + 1
            
            current_level = current_level + 1
        
        
            # Max Level reached... Extract also all spectra in the last level
        if np.size(all_pos, axis=0) == max_level + 1:
            
            for j in range(np.size(all_pos, axis=1)):
                if np.max(np.size(all_pos[-1][j])) > 1:
                    Extracted_Spectra = np.append(Extracted_Spectra, all_spectra[-1][j])
                    Extracted_end = np.append(Extracted_end, [max_level + 1, j])
                    CC = all_conc[-1][j]
                    Extracted_conc.append([CC > .5])
                    Extracted_pos.append(all_pos[-1][j])
                    all_count = all_count + 1
        
        Extracted_end = Extracted_end.reshape((round(len(Extracted_end)/2), 2))
        Extracted_Spectra = Extracted_Spectra.reshape((round(len(Extracted_Spectra) / np.size(x, axis=1)), np.size(x, axis=1)))
       
        return Extracted_Spectra, Extracted_end, Extracted_conc, Extracted_pos

    
dm3f = dm3.DM3("data8.dm3")
x= dm3f.imagedata

shapex = np.shape(x)
if len(shapex)==3:
    m,n,o =  np.shape(x)
    xi = x.copy()
    x = np.zeros((o*n, m))
    for i in range(o):
        x[i*n:(i+1)*n,:] = xi[:,:,i].T
else:
    x=x.T

max_iter=25

[Extracted_Spectra, Extracted_end, Extracted_conc, Extracted_pos] = HyperspectralSegmentationMCR_HIER_LLM.mcr_hier(x, max_iter) # appel de l'algorithme






