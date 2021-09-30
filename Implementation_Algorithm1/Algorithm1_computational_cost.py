#!/usr/bin/env python
# coding: utf-8

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

import random

import timeit

def remainder(a,n,m,y,k):
    
    return (a-np.dot(k[:len(y)],y))%m

def weight(y,k):
    return np.dot(np.diff(k,n=1)[:len(y)],y)

def flipped(n):
    if n==0:
        return 1
    elif n==1:
        return 0
    
def position1(y,r,k):
    pos=0
    if pos==r:
        return len(y)
    for i in np.arange(len(y))[::-1]:
        pos+=y[i]*(k[i+1]-k[i])
        if pos==r:
            return i
        
def position2(y,r,k):
    pos=0
    const=r-weight(y,k)-k[0]
    if pos==const:
        return 0
    for i in np.arange(len(y)):
        pos+=flipped(y[i])*(k[i+1]-k[i])
        if pos==const:
            return i+1

def deleted_seq1(p):
    return 0

def deleted_seq2(p):
    return 1


def I(p,b,y):
    return np.insert(y,p,b)

def dec_del(a,n,m,y,k):
    r=remainder(a,n,m,y,k)
    w=weight(y,k)
    if r<=w:
        p=position1(y,r,k)
        if p==None:
            return 'failure'
        b=deleted_seq1(p)
    else:
        p=position2(y,r,k)
        if p==None:
            return 'failure'
        b=deleted_seq2(p)
    return I(p,b,y)

def position(r,n,m,y,k):
    const=min([r,m-r])
    for i in np.arange(len(y)):
        if k[i]==const:
            return i

def Substituted(p,y):
     return np.insert(np.zeros(len(y)-1,dtype="int32"),p,1)^y

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

def dec_alg1(a,n,m,y,k):
    if len(y)==n:
        return dec_sub(a,n,m,y,k)
    elif len(y)==n-1:
        return dec_del(a,n,m,y,k)
    else:
        return 'failure'



x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[115]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[68]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[69]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[70]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[67]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[65]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[55]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[56]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[57]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[58]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[59]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[60]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[61]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[62]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[63]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[64]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:





# In[53]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[54]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[52]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[19]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[20]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[21]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[22]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[23]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[24]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[25]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[26]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[27]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[28]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[29]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[30]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[31]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[32]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[33]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[34]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[35]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[36]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[37]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[38]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[39]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[40]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[41]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[42]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[43]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[44]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[45]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[46]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[47]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[48]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[49]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[50]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[51]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[49]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[50]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[51]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[20]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[21]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[22]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[23]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[24]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[25]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[26]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[27]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[28]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[29]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[30]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[31]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[32]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[33]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n+1,n+2,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[34]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[35]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[36]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[37]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[38]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[39]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[40]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[41]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[42]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[43]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[44]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[45]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[46]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[108]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[109]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[110]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[111]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[112]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[113]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[114]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[75]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[76]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[77]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[78]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[79]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[80]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[81]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[82]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[83]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[84]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[85]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[86]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[87]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[88]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[89]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[90]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[91]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[94]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[95]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[96]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[97]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[98]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[99]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[107]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[100]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[106]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[104]:


x=np.arange(10000)+1
y=[(timeit.timeit(lambda: dec_alg1(0,n,2*n,random.choices([0,1],k=n),np.arange(n+1)+1), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[92]:


def random_binseq(n):
    rand=[]
    for i in np.arange(n) :
        rand+=[random.randint(0,1)]
    return rand


# In[93]:


dec_dr(0,4,9,[1,0,1],[1,3,8])


# In[30]:


dec_dr(0,6,20,[1,1,0,1,1,0],[1,2,3,8,9,10])


# In[31]:


dec_dr(0,5,20,[1,1,0,1,1,0],[1,2,3,8,9,10])


# In[39]:


dec_del(0,4,8,[1,0,1],[1,3])


# In[41]:


remainder(0,4,8,[1,0,1],[1,3])


# In[ ]:




