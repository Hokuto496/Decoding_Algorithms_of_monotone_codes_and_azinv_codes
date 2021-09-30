#!/usr/bin/env python
# coding: utf-8

import numpy as np

def azby_inverse(y):
    if len(y)%2==0:
        a=np.hstack((y[::2],y[::-2]))
    else:
        a=np.hstack((y[::2],y[len(y)-2::-2]))
    return a  

def count_inversion(y):
    return sum(sum(m<n for m in y[i+1:]) for i,n in enumerate(y))

def remainder(a,n,m,y,k=0):
    return (a-count_inversion(azby_inverse(y)))%m

def position(r,n,m,y,k=0):
    return len(y)-min([r,m-r])

def Substituted(p,y):
    return np.insert(np.zeros(len(y)-2,dtype="int32"),p-1,[1,1])^y

def dec_BAS(a,n,m,y,k=0):
    r=remainder(a,n,m,y)
    if r==0:
        return np.array(y)
    else:
        p= position(r,n,m,y)
        return Substituted(p,y)

    
dec_BAS(0,6,10,[0,1,0,1,1,1])




