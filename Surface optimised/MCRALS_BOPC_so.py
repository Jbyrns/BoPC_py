# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:33:12 2018

@author: jeffr
"""

""" Modify line 100 to select datafile to analyse, line 101 to modify the correct 
direction of the file (i.e. the observations must be attributed to the rows and 
the pixels to the columns),line 102 to modify the number of components and line 
103 to modify the number of iterations made by the algorithm (passing between c and s).
For any and all information please contact me at jeffrey.byrns@usherbrooke.ca""" 


import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy import linalg
import matplotlib.pyplot as plt
import dm3_lib as dm3


class HyperspectralSegmentation_MCR_ALS:
    
    @classmethod
    def mcr_als(cls,x, nb_c, max_iter=50):
        
        shapex = np.shape(x)
        if len(shapex)==3:
            m,n,o =  np.shape(x)
            xi = x.copy()
            c_t = np.zeros((o,nb_c,n))
            x = np.zeros((o*n, m))
            for i in range(o):
                x[i*n:(i+1)*n,:] = xi[:,:,i].T
        else:
            x=x.T

        x_n = np.sum(x, 1)
        x_n[x_n < 0.025*np.max(x_n)] = np.max(x_n)
        x = x.T / x_n
        x = x.T
        x.astype('float64')

        '''
        
        
        Updates: 
        1 - The issue was that it the spectral images contained many pixels with low or no signal (except random 
        noise), therefore when normalizing, the amplitude of those pixels would overcome the amplitude of the pixels 
        that had a true signal. So to keep the shape of the data, the normalization is now done in relation with the 
        maximum area under the curve. That way, the area of the most intense pixel will be 1 and the other ones will 
        simply be less than 1.
        
        2- I cannot only use the maximum of the sum because normalizing does allow to amplify features of low signal
        pixels. So line 48 serve as a filter to apply standard normalisation on pixel having sum > 0.05*max(sum(x,1) and 
        divide by max(sum(x,1)) for the pixels that are just random noise.
        
        3- Turns out, the reason the matrix became singular, was because one of the spectra was all negative (and was 
        then contrained to 0). The result was that the sum of that spectrum was 0 and thus making the matrix singular. 
        Adding a break allows to stop the iteration before it happens.
        '''


        # --- FAST INITIALIZATION ---
        # MiniBatchKmeans uses only 100 random observations at a time to speed up the process

        ctr_k = MiniBatchKMeans(n_clusters=nb_c).fit(x)

        s = ctr_k.cluster_centers_

        cnt = 0
        flag = True

        while flag:
            s_mem = s

            # Concentrations by linear regression between X and S
            c1 = s @ s.T
            c2 = linalg.inv(c1)
            c3 = x @ s.T
            c = c3 @ c2

            c[c < 0] = 0  # Modification to C - Non-negativity
            c_sum = np.sum(c, 1) # Modification to C - Closure
            c_sum[c_sum == 0] = 1  # This line prevent any division by 0
            c = c.T / c_sum
            c = c.T


            # Spectra by linear regression between X and C
            s1 = c.T @ c
            s2 = linalg.inv(s1)
            s3 = c.T @ x
            s = s2 @ s3
            s[s < 0] = 0
            c_mem = c
            cnt = cnt + 1

            if cnt == max_iter or np.min(np.sum(s, 1)) == 0:
                if np.min(np.sum(s, 1)) == 0:
                    s = s_mem
                    c = c_mem
                flag = False


        c = c.astype('float32')
        s = s.astype('float32')
        
        if len(shapex)==3:
            for i in range(n):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                c_t[:,:,i] = c[o*i:(1+i)*o,:]
            c=c_t

        
        
        return c, s
    

dm3f = dm3.DM3("data8.dm3")
x= dm3f.imagedata
nb_c =3  # nombre de composantes
nb_iter = 25

# nombre d'itérations

[c, s] = HyperspectralSegmentation_MCR_ALS.mcr_als(x,nb_c,nb_iter) # appel de l'algorithme

plt.plot(c)