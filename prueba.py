
# coding: utf-8

# # Analisis

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


# In[2]:


path = "../data"   #Path server 
#path = "./data" 

p_header="%s/header.txt" %(path)
p_data="%s/mobile_consume.csv" %(path)


# In[3]:


a = 45
b= 45
c = a + b/2
print("Loaded Data: ",c)

