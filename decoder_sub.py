#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def remainder(a,n,m,y,k):
     return (a-np.dot(k[0:len(y)],y))%m


# In[3]:


def position(r,n,m,y,k):
    for i in np.arange(len(y)):
        if k[i]==min([r,m-r]):
            return i


# In[4]:


def Substituted(p,y):
     return np.insert(np.zeros(len(y)-1,dtype="int32"),p,1)^y


# In[5]:


def dec_sub(a,n,m,y,k):
    r=remainder(a,n,m,y,k)
    if r==0:
        return np.array(y)
    else:
        p=position(r,n,m,y,k)
        return Substituted(p,y)







