#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


import numpy as np


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


import timeit


# In[15]:


import random


# In[16]:


def remainder(a,n,m,y,k):
     return (a-np.dot(k[0:len(y)],y))%m


# In[17]:


def position(r,n,m,y,k):
    const=min([r,m-r])
    for i in np.arange(len(y)):
        if k[i]==const:
            return i


# In[18]:


def Substituted(p,y):
     return np.insert(np.zeros(len(y)-1,dtype="int32"),p,1)^y


# In[19]:


def dec_sub(a,n,m,y,k):
    r=remainder(a,n,m,y,k)
    if r==0:
        return np.array(y)
    else:
        p=position(r,n,m,y,k)
        if p==None:
            return 'failure'
        else:
            return Substituted(p,y)


# In[20]:


dec_sub(0,6,20,[1,1,0,1,1,0],[1,2,3,8,9,10])


# In[22]:


dec_sub(1,6,20,[1,1,0,0,0,0],[1,2,3,8,9,10])


# In[22]:


remainder(0,6,20,[1,1,0,0,0,0],[1,2,3,8,9,10])


# In[25]:


position(17,6,20,[1,1,0,0,0,0],[1,2,3,8,9,10])


# In[23]:


dec_sub(1,6,20,[1,1,0,0,0,0],[1,2,3,8,9,10])


# In[24]:


dec_sub(1,6,20,[1,1,1,0,0,0],[1,2,3,8,9,10])


# In[25]:


dec_sub(0,6,20,[1,1,0,1,0,0],[1,2,3,8,9,10])


# In[44]:


x=np.arange(3000)+1
y=[(timeit.timeit(lambda: dec_rev(0,n,2*n,random_binseq(n),np.arange(n+1)+1), number=1)) for n in x]
plt.plot(x,y, '.')
plt.show()


# In[15]:


def random_binseq(n):
    rand=[]
    for i in np.arange(n) :
        rand+=[random.randint(0,1)]
    return rand


# In[ ]:




