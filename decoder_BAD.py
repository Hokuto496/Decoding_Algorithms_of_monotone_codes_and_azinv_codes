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
    return sum(sum(n>m for m in y[i+1:]) for i,n in enumerate(y))


# In[4]:


def evenflipped(y):
    return y^np.arange(len(y))%2


# In[5]:


def weight(y,k=0):
    a=evenflipped(y)
    return np.sum(a)


# In[6]:


def remainder(a,n,m,y,k=0):
    return (a-count_inversion(azby_inverse(y)))%m


# In[7]:


def position1(y,r,k=0):
    arr=np.array(evenflipped(y))
    for i in np.arange(len(arr)+1):
        if np.count_nonzero(arr[:len(arr)-i]==1)==r:
            return len(arr)-i


# In[8]:


def position2(y,r,k=0):
    arr=np.array(evenflipped(y))
    const=r-weight(y)-1
    for i in np.arange(len(arr)+1):
        if np.count_nonzero(arr[i:]==0)==const:
            return i


# In[9]:


def deleted_seq1(p):
    if p%2==0:
        return [0,1]
    else:
        return [1,0]


# In[10]:


def deleted_seq2(p):
    if p%2==0:
        return [1,0]
    else:
        return [0,1]


# In[11]:


def I(y,p,b):
    return np.insert(y,p,b)


# In[12]:


def dec_BAD(a,n,m,y,k=0):
    r=remainder(a,n,m,y)
    w=weight(y)
    if r<=w:
        p=position1(y,r)
        b=deleted_seq1(p)
    else: 
        p=position2(y,r)
        b=deleted_seq2(p)
    return I(y,p,b)


# In[13]:


dec_BAD(0,5,5,[1,1,0])


# In[14]:


dec_BAD(0,5,5,[1,0,0])


# In[15]:


dec_BAD(0,5,5,[1,0,1])


# In[70]:


dec_BAD(0,5,5,[0,1,0])


# In[71]:


dec_BAD(0,5,5,[0,0,0])


# In[72]:


dec_BAD(0,5,5,[1,1,1])


# In[73]:


dec_BAD(0,5,5,[0,1,1])


# In[74]:


dec_BAD(0,5,5,[0,0,1])


# In[ ]:





# In[ ]:




