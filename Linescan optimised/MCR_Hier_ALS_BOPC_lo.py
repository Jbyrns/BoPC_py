# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:50:40 2018

@author: jeffr
"""

import numpy as np

from scipy import linalg
from sklearn.cluster import MiniBatchKMeans




class HyperspectralSegmentationMCR_HIER_ALS:

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
        
        [c, s] = cls.mcr_als(x, 2, max_iter)

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
            
                        [c, s] = cls.mcr_als(x[cur_pos,:], 2, max_iter)
            
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

    @classmethod
    def mcr_als(cls,x, nb_c, max_iter=50):


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

        return c, s


x = np.loadtxt("data_2.txt", comments="#", delimiter=",", unpack=False) # Importation de la matrice
x = x.T
max_iter = 25  # nombre d'itérations

[Extracted_Spectra, Extracted_end, Extracted_conc, Extracted_pos] = HyperspectralSegmentationMCR_HIER_ALS.mcr_hier(x,max_iter) # appel de l'algorithme

