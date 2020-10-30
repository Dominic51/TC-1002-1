#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:21:06 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis

sp=pd.read_csv('/Users/diegovelazquez/Downloads/iris.csv')
sp['Tipo_Flor']=sp['Tipo_Flor'].replace(['Iris-versicolor','Iris-virginica','Iris-setosa'],[0,1,2])
data=sp.values
X= data[:,0:-1]
y=data[:,-1]

emb= FactorAnalysis(n_components=2)
X1t =emb.fit_transform(X,y)
plt.scatter(X1t[:,0],X1t[:,-1],c=y)
plt.title('Iris dataset FactorAnalysis')
plt.show()

emb= LinearDiscriminantAnalysis(n_components=2)
X2t =emb.fit_transform(X,y)
plt.scatter(X2t[:,0],X2t[:,-1],c=y)
plt.title('Iris dataset LinearDiscriminant')
plt.show()

emb=NeighborhoodComponentsAnalysis(n_components=2)
X3t =emb.fit_transform(X,y)
plt.scatter(X3t[:,0],X3t[:,-1],c=y)   
plt.title('Iris dataset Neighborhood')
plt.show()

emb= Isomap(n_components=2)
X4t =emb.fit_transform(X,y)
plt.scatter(X4t[:,0],X4t[:,-1],c=y)   
plt.title('Iris dataset Isomap')
plt.show()


emb=MDS(n_components=2)
x5t= emb.fit_transform(X,y)
plt.scatter(x5t[:,0],x5t[:,1],c=y)
plt.title('Iris dataset MDS')
plt.show()

def dE(datos1, datos2, leng):
dist = 0
for i in range(leng):
dist += np.square(datos1[i] - datos2[i])
return np.sqrt(dist)

def multknn(dfTraining, dfPredic,k):
y=[]
ll=dfPredic.values.tolist()

for i in range(dfPredic.shape[0]):
dato=pd.DataFrame(ll[i])
result,neigh = knn(dfTraining, dato, k)
y.append(result)
return y
