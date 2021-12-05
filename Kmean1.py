# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 02:10:15 2021

@author: Mi8a
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from knead import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('spambase.csv')

X = dataset.drop('class', axis=1)
y = dataset['class']

wcss_list= []  #Initializing the list for the values of WCSS  
  
#Using for loop for iterations from 1 to 10.  
for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
    kmeans.fit(X)  
    wcss_list.append(kmeans.inertia_)  

features, true_labels = make_blobs(n_samples=200,centers=3,cluster_std=2.75,random_state=42)
features[:5]
true_labels[:5]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]

kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=42)
kmeans.fit(scaled_features)
kmeans.inertia_
kmeans.cluster_centers_
kmeans.n_iter_
kmeans.labels_[:5]

kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42}
   
# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
   kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
   kmeans.fit(scaled_features)
   sse.append(kmeans.inertia_)
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()   

kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")

kl.elbow


