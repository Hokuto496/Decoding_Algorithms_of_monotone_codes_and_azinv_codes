#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import random


# In[5]:


import timeit


# In[6]:


def azby_inverse(y):
    if len(y)%2==0:
        a=np.hstack((y[::2],y[::-2]))
    else:
        a=np.hstack((y[::2],y[len(y)-2::-2]))
    return a  


# In[7]:


def reverse(k):
    return np.arange(len(k))[::-1]+1


# In[8]:


def inversion_number(k):
    return np.dot(k,reverse(k))-(np.sum(k)*(np.sum(k)+1))//2


# In[9]:


def evenflipped(y):
    return y^np.arange(len(y))%2


# In[10]:


def weight(y,k=0):
    a=evenflipped(y)
    return np.sum(a)


# In[11]:


def remainder(a,n,m,y,k=0):
    return (a-inversion_number(azby_inverse(y)))%m


# In[12]:


def flipped(b):
    if b==0:
        return 1
    elif b==1:
        return 0


# In[13]:


def position1(y,r,k=0):
    arr=np.array(evenflipped(y))
    pos=np.count_nonzero(arr==1)
    if pos==r:
            return len(arr)
    for i in np.arange(len(arr))[::-1]:
        pos-=arr[i]
        if pos==r:
            return i


# In[14]:


def position2(y,r,k=0):
    arr=np.array(evenflipped(y))
    pos=np.count_nonzero(arr==0)
    const=r-weight(y)-1
    if pos==const:
            return 0
    for i in np.arange(len(arr))+1:
        pos-=flipped(arr[i-1])
        if pos==const:
            return i


# In[15]:


def deleted_seq1(p):
    if p%2==0:
        return [0,1]
    else:
        return [1,0]


# In[16]:


def deleted_seq2(p):
    if p%2==0:
        return [1,0]
    else:
        return [0,1]


# In[17]:


def I(y,p,b):
    return np.insert(y,p,b)


# In[18]:


def dec_BAD(a,n,m,y,k=0):
    r=remainder(a,n,m,y)
    w=weight(y)
    if r<=w:
        p=position1(y,r)
        if p==None:
            return'failure'
        b=deleted_seq1(p)
    else: 
        p=position2(y,r)
        if p==None:
            return 'failure'
        b=deleted_seq2(p)
    return I(y,p,b)


# In[19]:


dec_BAD(0,5,5,[1,1,0])


# In[20]:


dec_BAD(0,5,5,[1,0,0])


# In[21]:


dec_BAD(0,5,5,[1,0,1])


# In[22]:


dec_BAD(0,5,5,[0,1,0])


# In[23]:


dec_BAD(0,5,5,[0,0,0])


# In[24]:


dec_BAD(0,5,5,[1,1,1])


# In[25]:


dec_BAD(0,5,5,[0,1,1])


# In[26]:


dec_BAD(50,5,100,[0,1])


# In[19]:


def position(r,n,m,y,k=0):
    return len(y)-min([r,m-r])


# In[20]:


def Reversed(p,y):
    return np.insert(np.zeros(len(y)-2,dtype="int32"),p-1,[1,1])^y


# In[21]:


def dec_BAR(a,n,m,y,k=0):
    r=remainder(a,n,m,y)
    if (np.all(y==np.ones(len(y))))|(np.all(y==np.zeros(len(y)))):
        return 'failure'
    elif r==0:
        return np.array(y)
    else:
        p= position(r,n,m,y)
        if p==None:
            return 'failure'
        else:
            return Reversed(p,y)


# In[22]:


def dec_alg2(a,n,m,y,k=0):
    if len(y)==n:
        return dec_BAR(a,n,m,y,k)
    elif len(y)==n-2:
        return dec_BAD(a,n,m,y,k)
    else:
        return 'failure'


# In[105]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[108]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[109]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[110]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[111]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[112]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[113]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[114]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[115]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[116]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[117]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[118]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[119]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[120]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[121]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[122]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[123]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[124]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[125]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[126]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[127]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[128]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[129]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[130]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[131]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[132]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[133]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[134]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[135]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[136]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[137]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[138]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[139]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[140]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[141]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[142]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[143]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[144]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[145]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[146]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[147]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[148]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[149]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[150]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[151]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[107]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[102]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[103]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[104]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[89]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[99]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[100]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[101]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[90]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[91]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[92]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[93]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[94]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[95]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[96]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[97]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[98]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[76]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[77]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[78]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[79]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[80]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[81]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[82]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[83]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[84]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[85]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[86]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[87]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:





# In[75]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[74]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[73]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[23]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[24]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[25]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[26]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[27]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[28]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[29]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[30]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[31]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[32]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[33]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[34]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[35]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[36]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[37]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[38]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[39]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[40]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[41]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[42]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[43]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[44]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[45]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[46]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[47]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[48]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[49]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[50]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[51]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[52]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[53]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[54]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[55]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[56]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[57]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[58]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[59]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[60]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[61]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[62]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[63]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[64]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[65]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[66]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[67]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[68]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[69]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[70]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[71]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[72]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[55]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[56]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[57]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[24]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[25]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[26]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[27]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
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





# In[28]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[29]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[30]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[31]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[32]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[33]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[34]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[35]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[36]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[37]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[38]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[39]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[40]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[41]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[42]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[43]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[44]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[45]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[46]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[47]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[48]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[49]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[50]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[51]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[52]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[53]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[54]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[152]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[153]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[154]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[155]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[156]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[157]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[158]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[159]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[160]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[161]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[162]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[163]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[164]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[165]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[166]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[167]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[168]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[169]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[170]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[172]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[173]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[174]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[175]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[176]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[177]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[178]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[179]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[180]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[180]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[180]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[180]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[180]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[180]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[180]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=3)/3) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:





# In[181]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[182]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[183]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[187]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[188]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[189]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[190]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[191]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[193]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[194]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[195]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[192]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[23]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[24]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[25]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[26]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[27]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[28]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[39]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[37]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[38]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[40]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[41]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[43]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[42]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[44]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n+2,n+2,random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[184]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[185]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[186]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[29]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[30]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[31]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[32]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[33]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[34]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[35]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[36]:


x=np.arange(10000)+2
y=[(timeit.timeit(lambda: dec_alg2(0,n,2*(n-1),random.choices([0,1],k=n)), number=10)/10) for n in x]
plt.plot(x,y, '.')
plt.title('Decoding time')
plt.ylabel('Running time (seconds)')
plt.xlabel('Code-length')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[171]:


def random_binseq(n):
    rand=[]
    for i in np.arange(n) :
        rand+=[random.randint(0,1)]
    return rand


# In[37]:


x=np.arange(3000)+1
y=[(timeit.timeit(lambda: dec_BAD(0,n+1,n+2,random_binseq(n),np.arange(n+1)+1),number=1)) for n in x]
plt.plot(x,y, '.')
plt.show()


# In[24]:


weight([0,0,0])


# In[28]:


weight(np.zeros(3,dtype='int32'))


# In[27]:


np.zeros(3,dtype='int32')


# In[33]:


dec_BADR(0,5,5,[1,1,0])


# In[34]:


dec_BADR(0,4,5,[1,1,0])


# In[35]:


dec_BADR(0,4,2,[1,1,0])


# In[36]:


dec_BADR(0,6,10,[0,1,0,0,0,0])


# In[66]:


2*3


# In[27]:


random.choices([0,1],k=3)


# In[20]:


def reverse(y):
    return np.arange(len(y))[::-1]+1


# In[ ]:




