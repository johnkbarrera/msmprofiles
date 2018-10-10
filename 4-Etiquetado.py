
# coding: utf-8

# # JOIN GROUPS

# In[1]:


import numpy as np
import datetime
from datetime import date
import json
import pylab
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from sklearn.preprocessing import normalize


# In[2]:


import configparser
Config = configparser.ConfigParser()
Config.read("Config.conf")


# In[3]:


def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


# # RESULTADOS

# ## Unimos los resultados por cada mccg

# Juntamos los datos originales con sus etiquetas de cluster individuale y cluster colectivo correspondientes.
# 
# <div>Utulizamos solo 2 variables para el ahorro de memoria. </div>

# In[4]:


# choosen files

footprint = ConfigSectionMap("f2")['footprints']
individual_clusters = ConfigSectionMap("f3")['ic_cluster']
individual_labels = ConfigSectionMap("f3")['ic_label'] 
collective_clusters = ConfigSectionMap("f3")['cc_cluster']
collective_labels = ConfigSectionMap("f3")['cc_label'] 
                                                     


# In[5]:


# read file collective_labels
b = pd.read_csv(collective_labels, sep=";", header=0, index_col=False, low_memory=False)[['CO_ID', 'INDIVIDUAL_CLUSTER', 'COLLECTIVE_CLUSTER']] 
# read file individual_labels
a = pd.read_csv(individual_labels, sep=";", header=0, index_col=False, dtype={'WEEK': str,'YEAR': str}, low_memory=False)[['CO_ID', 'YEAR', 'WEEK', 'INDIVIDUAL_CLUSTER']]


# In[6]:


b = pd.merge(a, b, on=['CO_ID', 'INDIVIDUAL_CLUSTER'])
del(a)


# In[7]:


a = pd.read_csv(footprint, sep=",", header=0, dtype={'YEAR': str,'WEEK': str}, low_memory=False)   ## read file


# In[8]:


a = pd.merge(a, b, on=['CO_ID','YEAR', 'WEEK'])
del(b)


# In[9]:


len(a)


# In[10]:


path_res = ConfigSectionMap("f4")['union']
a.to_csv(path_res,index=False)
del a
print('Done')

