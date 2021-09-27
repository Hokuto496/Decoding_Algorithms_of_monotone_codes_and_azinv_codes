#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#元の符号語とのズレの大きさ
def remainder(a,n,m,y,k):
    return (a-np.dot(k[:len(y)],y))%m 


# In[3]:


#ズレの大きさの基準値
def weight(y,k):
    return np.dot(np.diff(k,n=1)[:len(y)],y)


# In[4]:


#挿入位置その１
def position1(r,y,k):
    pos=0
    if pos==r:
        return len(y)
    for i in np.arange(len(y))[::-1]:
        pos+=y[i]*(k[i+1]-k[i])
        if pos==r:
            return i


# In[5]:


#挿入位置その２
def flipped(n):
    if n==0:
        return 1
    elif n==1:
        return 0

def position2(r,y,k):
    pos=0
    const=r-weight(y,k)-k[0]
    if pos==const:
        return 0
    for i in np.arange(len(y)):
        pos+=flipped(y[i])*(k[i+1]-k[i])
        if pos==const:
            return i+1


# In[6]:


#挿入文字その１
def deleted_seq1(p):
    return 0


# In[7]:


#挿入文字その２
def deleted_seq2(p):
    return 1


# In[8]:


#系列yのp番目にbを挿入する写像
def ins(p,b,y):
    return np.insert(y,p,b)


# In[9]:


#アルゴリズム１の削除誤り訂正をする部分
def dec_del(a,n,m,y,k):
    r=remainder(a,n,m,y,k)
    w=weight(y,k)
    if r<=w:
        p=position1(r,y,k)
        if p==None:
            return 'failure'
        b=deleted_seq1(p)
    else:
        p=position2(r,y,k)
        if p==None:
            return 'failure'
        b=deleted_seq2(p)
    return ins(p,b,y)


# In[10]:


"""
符号語の長さn=5，整数a=2，1以上の整数m=9，数列k=(1,2,3,6,8)，系列y=0111．
このときの単調増加符号M_{a,m,k}(n)={01000,11001,01110,00101,11111}の元01110に
復号されるか確認する．
"""
dec_del(2,5,9,[0,1,1,1],[1,2,3,6,8])


# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


dec_del(0,4,9,[1,0,1],[1,3,6,8])


# In[ ]:





# In[ ]:





# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

import random

import timeit


# In[71]:


dec_del(72,4,100,[1,0,1],[1,3,6,8])


# In[72]:


remainder(0,4,8,[1,0,1],[1,3,6,8])


# In[73]:


weight([1,0,1],[1,3,6,8])


# In[77]:


position1(1,[1,0,1],[1,3,6,8])==None


# In[76]:


dec_del(0,4,9,[0,1,0],[1,3,6,8])


# In[58]:


def random_binseq(n):
    rand=[]
    for i in np.arange(n) :
        rand+=[random.randint(0,1)]
    return rand


# In[59]:


x=np.arange(3000)+1
y=[(timeit.timeit(lambda: dec_del(0,n+1,n+2,random_binseq(n),np.arange(n+1)+1),number=1)) for n in x]
plt.plot(x,y, '.')
plt.show()


# In[ ]:




