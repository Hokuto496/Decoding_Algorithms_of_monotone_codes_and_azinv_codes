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


def dec_BAR(a,n,m,y,k=0):
    r=remainder(a,n,m,y)
    if r==0:
        return np.array(y)
    else:
        p= position(r,n,m,y)
        return Substituted(p,y)


