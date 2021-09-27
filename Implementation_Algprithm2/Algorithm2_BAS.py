#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def azby_inverse(y):
    if len(y)%2==0:
        a=np.hstack((y[::2],y[::-2]))
    else:
        a=np.hstack((y[::2],y[len(y)-2::-2]))
    return a  


# In[3]:


def count_inversion(y):
    return sum(sum(m<n for m in y[i+1:]) for i,n in enumerate(y))


# In[4]:


def remainder(a,n,m,y,k=0):
    return (a-count_inversion(azby_inverse(y)))%m


# In[5]:


def position(r,n,m,y,k=0):
    return len(y)-min([r,m-r])


# In[6]:


def Substituted(p,y):
    return np.insert(np.zeros(len(y)-2,dtype="int32"),p-1,[1,1])^y


# In[7]:


def dec_BAS(a,n,m,y,k=0):
    r=remainder(a,n,m,y)
    if r==0:
        return np.array(y)
    else:
        p= position(r,n,m,y)
        return Substituted(p,y)


# In[16]:


get_ipython().run_line_magic('timeit', 'dec_BAR(0,6,10,[0,1,0,0,0,0])')


# In[17]:


get_ipython().run_line_magic('timeit', 'dec_BAR(0,6,10,[1,0,0,0,0,0])')


# In[9]:


dec_BAS(0,6,10,[0,1,0,1,1,1])


# In[11]:


dec_BAS(0,6,10,[0,1,1,0,1,1])


# In[ ]:




