# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:58:32 2018

@author: jeffr
"""


import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import hdbscan
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist

x = np.loadtxt("data_2.txt", comments="#", delimiter=",", unpack=False) # Importation de la matrice
x = x.T
nb_c =7  # nombre de composantes





x_sum = np.asarray([np.sum(x, axis=1)]).T
x_norm = x/x_sum
x_norm = np.nan_to_num(x_norm)
x_raw = x  # note that lambda is a function in python, lada replaces lambda as not to overwrite it. Also lada is a shitty russian car manufacturer, look it up!

""" Kmeans clustering as a comparisson """
ckm = KMeans(n_clusters=nb_c).fit(x_norm).labels_

skm = np.zeros(( np.size(x_norm, axis = 1),nb_c))
for j in range(0,nb_c):
    filter1 = ckm==j
    spec = np.mean(x_norm[filter1,:],axis =0).T
    skm[:,j-1] = spec.T
    
   

skm = skm.T
ckmr = x_norm @ skm.T @ np.linalg.inv(skm @ skm.T)
plt.subplot(5,1,1)
plt.plot(ckmr)
plt.title(" Kmeans clustering as a comparisson ")
plt.xlabel('pixel')
plt.ylabel('composition')

""" agglomerative clustering Ward """
cacw= AgglomerativeClustering(n_clusters=nb_c, linkage= 'ward').fit(x_norm).labels_
sacw = np.zeros(( np.size(x_norm, axis = 1),nb_c))
for j in range(0,nb_c):
    filter1 = cacw==j
    spec = np.mean(x_norm[filter1,:],axis =0).T
    sacw[:,j-1] = spec.T
    
   

sacw = sacw.T
cacwr = x_norm @ sacw.T @ np.linalg.inv(sacw @ sacw.T)
plt.subplot(5,1,2)
plt.plot(cacwr)
plt.title(" agglomerative clustering Ward ")
plt.xlabel('pixel')
plt.ylabel('composition')

""" agglomerative clustering Complete """
cacc= AgglomerativeClustering(n_clusters=nb_c, linkage= 'complete').fit(x_norm).labels_
sacc = np.zeros(( np.size(x_norm, axis = 1),nb_c))
for j in range(0,nb_c):
    filter1 = cacc==j
    spec = np.mean(x_norm[filter1,:],axis =0).T
    sacc[:,j-1] = spec.T
    
   

sacc = sacc.T
caccr = x_norm @ sacc.T @ np.linalg.inv(sacc @ sacc.T)
plt.subplot(5,1,3)
plt.plot(caccr)
plt.title(" agglomerative clustering Complete ")
plt.xlabel('pixel')
plt.ylabel('composition')

""" agglomerative clustering Average """
caca= AgglomerativeClustering(n_clusters=nb_c, linkage= 'average').fit(x_norm).labels_
saca = np.zeros(( np.size(x_norm, axis = 1),nb_c))
for j in range(0,nb_c):
    filter1 = caca==j
    spec = np.mean(x_norm[filter1,:],axis =0).T
    saca[:,j-1] = spec.T
    
   

saca = saca.T
cacar = x_norm @ saca.T @ np.linalg.inv(saca @ saca.T)
plt.subplot(5,1,4)
plt.plot(cacar)
plt.title(" agglomerative clustering Average ")
plt.xlabel('pixel')
plt.ylabel('composition')


""" HDBSCAN clustering"""

for mcs in range(2,10):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
    cluster_labels = clusterer.fit_predict(x_norm)
    
    number_of_components = len(np.unique(cluster_labels))
    mcs+=1

    if number_of_components == nb_c+1: #The addition of a component is necessary since this method clusters all noise in its -1 cluster label
        break  
  
sHDBSCAN = np.zeros(( np.size(x_norm, axis = 1),nb_c))

for j in range(nb_c):
    filter1 = cluster_labels==j
    spec = np.mean(x_norm[filter1,:],axis =0).T
    sHDBSCAN[:,j-1] = spec.T
       

sHDBSCAN = sHDBSCAN.T
cHDBSCAN = x_norm @ sHDBSCAN.T @ np.linalg.inv(sHDBSCAN @ sHDBSCAN.T)
plt.subplot(5,1,5)
plt.plot(cHDBSCAN)
plt.title(" HDBSCAN clustering ")
plt.xlabel('pixel')
plt.ylabel('composition')

plt.show()