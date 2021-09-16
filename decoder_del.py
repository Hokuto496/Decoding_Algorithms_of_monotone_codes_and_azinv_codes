#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def remainder(a,n,m,y,k):
    return (a-np.dot(k[:len(y)],y))%m


# In[3]:


def weight(y,k):
    return np.dot(np.diff(k,n=1)[:len(y)],y)


# In[4]:


def position1(y,r,k):
    for i in np.arange(len(y)):
        if np.dot(np.diff(k,n=1)[len(y)-1-i:len(y)],y[len(y)-1-i:])==r:
            return len(y)-1-i


# In[19]:


def position2(y,r,k):
    b=np.ones(len(y),dtype="int32")^y
    const=r-weight(y,k)-k[0]
    for i in np.arange(len(b)):
        if np.dot(np.diff(k,n=1)[:i],b[:i])==const:
            return i 


# In[20]:


def deleted_seq1(p):
    return 0


# In[21]:


def deleted_seq2(p):
    return 1


# In[22]:


def I(y,p,b):
    return np.insert(y,p,b)


# In[25]:


def dec_del(a,n,m,y,k):
    r=remainder(a,n,m,y,k)
    w=weight(y,k)
    if r<=w:
        p=position1(y,r,k)
        b=deleted_seq1(p)
    else:
        p=position2(y,r,k)
        b=deleted_seq2(p)
    return I(y,p,b)


# In[26]:


dec_del(0,4,9,[1,0,1],[1,3,6,8])


# In[27]:


dec_del(0,4,9,[0,1,0],[1,3,6,8])


# In[ ]:





# In[ ]:





# In[ ]:




