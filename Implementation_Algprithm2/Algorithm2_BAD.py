#!/usr/bin/env python
# coding: utf-8

import numpy as np


#元の符号語とのズレ
def azby_inverse(y):
    if len(y)%2==0:
        a=np.hstack((y[::2],y[::-2]))
    else:
        a=np.hstack((y[::2],y[len(y)-2::-2]))
    return a  

def reverse(k):
    return np.arange(len(k))[::-1]+1

def inversion_number(k):
    return np.dot(k,reverse(k))-(np.sum(k)*(np.sum(k)+1))//2

def remainder(a,n,m,y,k=0):
    return (a-inversion_number(azby_inverse(y)))%m


#ズレの大きさの基準値
def evenflipped(y):
    return y^np.arange(len(y))%2

def weight(y,k=0):
    a=evenflipped(y)
    return np.sum(a)

#挿入位置その１
def position1(y,r,k=0):
    arr=np.array(evenflipped(y))
    pos=np.count_nonzero(arr==1)
    if pos==r:
            return len(arr)
    for i in np.arange(len(arr))[::-1]:
        pos-=arr[i]
        if pos==r:
            return i

#挿入位置その２
def flipped(b):
    if b==0:
        return 1
    elif b==1:
        return 0
    
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

#挿入文字その１
def deleted_seq1(p):
    if p%2==0:
        return [0,1]
    else:
        return [1,0]

#挿入文字その２
def deleted_seq2(p):
    if p%2==0:
        return [1,0]
    else:
        return [0,1]

#系列yのp番目にbを挿入する写像
def ins(y,p,b):
    return np.insert(y,p,b)

#アルゴリズム２のBAD誤り訂正をする部分
def dec_BAD(a,n,m,y,k=0):
    r=remainder(a,n,m,y)
    w=weight(y)
    if r<=w:
        p=position1(y,r)
        b=deleted_seq1(p)
    else: 
        p=position2(y,r)
        b=deleted_seq2(p)
    return ins(y,p,b)

"""
符号語の長さn=5，整数a=0，1以上の整数m=5，系列y=011
このときのAzinv符号A_{a,m}(n)={01000,01010,01011,01111,10110,10001}の元01011
に復号されるか確認する．
"""
dec_BAD(0,5,5,[0,1,1])




