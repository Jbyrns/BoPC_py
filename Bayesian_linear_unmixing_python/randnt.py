# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:47:15 2018

@author: jeffr
"""
import numpy as np

class randnt:
    

    @classmethod
    def randnt(m,s,N):
        
        """RPNORM    Random numbers from the positive normal distribution.
           RPNORM(N,M,S) is an N-by-1 vector with random entries, generated
           from a positive normal distribution with mean M and standard
           deviation S.
        
         (c) Vincent Mazet, 06/2005
         Centre de Recherche en Automatique de Nancy, France
         vincent.mazet@cran.uhp-nancy.fr
        
         Reference:
         V. Mazet, D. Brie, J. Idier, "Simulation of Positive Normal Variables
         using several Proposal Distributions", IEEE Workshop Statistical
         Signal Processing 2005, july 17-20 2005, Bordeaux, France. """
        
        
        
        if s<0:
            print('Standard deviation must be positive.')
            return-1
        if N<=0:
            print('N is wrong.')
            return-1
            
        Tindcand = [];
        x = [];     # Output vector
        NN = N;
        
        """ Intersections """
        A  = 1.136717791056118
        mA = (1-A^2)/A*s
        mC = s * np.sqrt(np.pi/2)
        
        while len(x)<NN:
	
            if m < mA:      # 4. Exponential distribution
                a = (-m + np.sqrt(m**2+4*s**2)) / 2 / s**2
                z = -np.log(1-np.random.random((N,1)))/a
                rho = np.exp( -(z-m)**2/2/s**2 - a*(m-z+a*s**2/2) )
            elif m <= 0:  # 3. Normal distribution truncated at the mean
                """ equality because 3 is faster to compute than the 2 """
                z = abs(np.random.random((N,1)))*s + m
                rho = (z>=0)
            elif m < mC:  # 2. Normal distribution coupled with the uniform one
                r = (np.random.random((N,1)) < m/(m+np.sqrt(np.pi/2)*s))
                u = np.random.random((N,1))*m
                g = abs(np.random.random((N,1))*s) + m
                z = r*u + (1-r)*g
                rho = r*np.exp(-(z-m)**2/2/s**2) + (1-r)*np.ones((N,1))
            else:           # 1. Normal distribution
                z = np.random.random((N,1))*s + m
                rho = (z>=0)
           
            
            """ Accept/reject the propositions """
            reject = (np.random.random((N,1)) > rho)
            z = np.delete(z,reject,0)
            if len(z)>0:
                x = [[x], [z]]
            N = N-len(z);
        return x
            