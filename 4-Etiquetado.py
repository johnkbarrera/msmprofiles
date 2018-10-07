
# coding: utf-8

# # JOIN GROUPS

# In[9]:


import numpy as np
import datetime
from datetime import date
import json
import pylab
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from sklearn.preprocessing import normalize


# # RESULTADOS

# ## Unimos los resultados por cada mccg

# Juntamos los datos originales con sus etiquetas de cluster individuale y cluster colectivo correspondientes.
# 
# <div>Utulizamos solo 2 variables para el ahorro de memoria. </div>

# In[10]:


path='./resultados' 

file="%s/U" %(path) 
footprint="%s.footprint" %(file)
individual_clusters="%s.individual_footprint.clusters" %(file)
individual_labels="%s.individual_footprint.labels" %(file)   
collective_clusters="%s.collective_footprint.clusters" %(file)
collective_labels="%s.collective_footprint.labels" %(file)


# In[11]:


# read file collective_labels
b = pd.read_csv(collective_labels, sep=";", header=0, index_col=False, low_memory=False)[['CO_ID', 'INDIVIDUAL_CLUSTER', 'COLLECTIVE_CLUSTER']] 
# read file individual_labels
a = pd.read_csv(individual_labels, sep=";", header=0, index_col=False, dtype={'WEEK': str,'YEAR': str}, low_memory=False)[['CO_ID', 'YEAR', 'WEEK', 'INDIVIDUAL_CLUSTER']]


# In[12]:


b = pd.merge(a, b, on=['CO_ID', 'INDIVIDUAL_CLUSTER'])
del(a)


# In[13]:


a = pd.read_csv(footprint, sep=",", header=0, dtype={'YEAR': str,'WEEK': str}, low_memory=False)   ## read file


# In[14]:


a = pd.merge(a, b, on=['CO_ID','YEAR', 'WEEK'])
del(b)


# In[15]:


len(a)


# In[16]:


path_res='%s/Results.csv' %(path)
a.to_csv(path_res,index=False)
del a
print('Done')

