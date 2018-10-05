
# coding: utf-8

# # CONSTRUCCIÓN DE LOS "FOOTPRINTS"

# Construimos los footprints temporales de cada cliente a partir de sus transacciones, de tal manera que agrupamos en un intervalo de d días.
# 
# 
# Dado:
# 
#     Sesion de TX
#         s = {cliente, timestamp, Monto}
#     
#     Cada cliente tiene una secuencia de sesiones de TX (S)
#         S = { s1, s2 , s3, ..., sn}                n: # de sesiones de cada cliente
#         
#         

# In[1]:


# LIBRERIAS
import numpy as np
import datetime
from datetime import date
import json
import pylab
import pandas as pd
import matplotlib.pyplot as plt
import os, sys


# ## Preparacion de datos

# ### Cargando datos

# In[2]:


#path = "../data"   #Path server 
path = "./data" 
p_header="%s/header.txt" %(path)
p_data="%s/mobile_consume.csv" %(path)


# In[3]:


header = pd.read_csv(p_header)
data = pd.read_csv(p_data , header = None)
data.columns = list(header)


# ### Preparacion de datos

# In[4]:


data['AÑO'] = data['F_TRAFICO'].apply(lambda fecha: float(fecha[6:]))


# In[5]:


data.head()


# ## Definicion de variables

# ### Clientes

# In[6]:


# DEFINIMOS LA LISTA DE CLIENTES
clientes =  data.groupby('CO_ID').CO_ID.count().index
clientes


# # FOOTPRINT PARA CADA MCCG

# ## Unidad de TXs temporales (U)

# Resume un conjunto de TXs en un periodo de tiempo **“tau”**.
# <div>Los footprints que creamos será la agrupación de todas las transacciones en "tau" de cada cliente, y representará una unidad de comportamiento con 3 dimensiones.</div> 
# 
# 
# Donde:
# 1.  **tau** = 1 semana
# 2.  dimensión 1:    **d**   = 7 días
# 3.  dimensión 2:	**t**   = 4 turnos	 
# 4.  dimensión 3:    **m**   = # de mccgs en nuestra data.
# 
#     
# Para cada cliente:
#     
#     Entrada:   S = {s1, ..., sn}
#     Salida:    SS = {U(1), U(2), U(3), ...,U(m)}     donde m <=n

# ### Funciones de apoyo

# Definimos funcion para generar los footprint (U) de un usuario, con los parametros:
# 
# 1. <div> **user**:  Dataset original filtrado para un cliente determinado</div> 
# 2. <div> **monto**: Indica si los footprint seran generados con la suma de los el numero de TXs (False) o la suma de los montos por cada TX (True)</div> 

# In[7]:


def procesar_u(user, tipo_eth = False):    
    uid=list(user['CO_ID'])[0]              # Cliente_id
    years = set(list(user['AÑO']))              # Lista los años en que se tiene TXs registradas
    anni = {year:{} for year in list(years)}    # definimos anni como una lista 
    
    # para cada fila, es decir, cada TXs del cliente)
    for dat in  range(0,len(user)):
        año = int(user.iloc[dat]['AÑO'])
        fecha = user.iloc[dat]['F_TRAFICO']
        fecha = pd.to_datetime(fecha, format='%d/%m/%Y', errors='coerce')
        mes = fecha.month
        dia = fecha.day        
        turn = user.iloc[dat]['HORA']
        
        week=str(datetime.datetime(año,mes,dia).isocalendar()[1])
        if len(week)==1:
            week = '0' + week
        weekday=datetime.datetime(año,mes,dia).weekday()
        
        # Si la semana no existe en el año
        if not(week in anni[año]):
            anni[año][week] = {}
        # Si el billcycle no existe en la semana y año
        if not (weekday in anni[año][week]):
            anni[año][week][weekday]={}  #NUMERO DE MCCGs VARIABLES
        # Si el turno no existe en el mccg,semana y año
        
        anni[año][week][weekday][turn]=list(user.iloc[dat,6:-1]) 
                
    return uid,anni


# ### Procesando U

# Generamos en formato json el footprint de cada cliente
# donde el los índices son: **cliente_id**, **año**, **semana**, **mccg**, **turno** conteniendo un **array[0,...,6]** con los dias.
# 
# 

# In[8]:


##################################################
#        Procesando U de cada CLIENTE
##################################################
    
profiles={}           # Creamos lista de prefiles
contador=0 
print("Number of rows "+str(len(data))) 

# Para cada cliente
for cliente in clientes:
    cliente_i= data[data['CO_ID'] == cliente]       # filtramos dataset solo para el cliente i
    results=procesar_u(cliente_i, tipo_eth=False)          # procesamos u del usuario i
    profiles[results[0]]=results[1]                     # cargamos lista de indice "uid" con la data del cliente(json)
    contador += 1
    if contador % 1000 == 1:
        print("vamos en el ",contador)


# In[9]:


# profiles


# Creamos la cabecera dinámica donde se guardaran todos los footprints generados

# In[10]:


cabecera = 'CO_ID,YEAR,WEEK,PROFILE_ID,SIZE'


# In[11]:


for i in range(7):      # numero de dias
    for j in range(4):                # numero de turnos
        for k in range(38):            # numero de planes
            cabecera = cabecera+','+'D'+str(i)+'T'+str(j)+'P'+str(k)
cabecera = cabecera+'\n'


# In[12]:


outfile='./resultados/U'           # Indicamos archivo de salida 
individual_footprint="%s.footprint" %(outfile)
fw=open(individual_footprint,'w')  

fw.write(cabecera)                    # Escribimos la cabecera


# In[13]:



# Para cada uid (cliente)
footprints=0
for uid in profiles:   
    profile_id=0
    # En cada año
    for year in profiles[uid]:       
        # Por cada semana
        for week in profiles[uid][year]:    
                             
            temp=np.zeros(7*4*38) 
            # Por cada semana
            for weekday in profiles[uid][year][week]:
                temp2=np.zeros(4*38) 
                # Por cada turno
                for turno in profiles[uid][year][week][weekday]:                        
                    # print(uid,year,week,weekday,turno,len(profiles[uid][year][week][weekday][turno]))
                    temp2[turno*38:(turno+1)*38] = profiles[uid][year][week][weekday][turno]
                temp[weekday*len(temp2):(weekday+1)*len(temp2)] = temp2
          
            # Escribimos los datos del primer comportamiento (Tensor)    
            txt = ''+str(uid)+','+str(year)+','+str(week)+','+str(profile_id)+','+str(sum(temp))
            for i in range(len(temp)):
                txt = txt +','+str(temp[i])
            fw.write(txt +'\n')

            profile_id += 1   
            footprints += 1  
            
    fw.flush()
fw.close()               
print ("number of footprint: "+str(footprints))


# In[14]:


file='./resultados/U' 
footprint="%s.footprint" %(file)
data = pd.read_csv(footprint)


# In[15]:


print("Done")

