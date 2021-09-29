import numpy as np

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


def Reversed(p,y):
     return np.insert(np.zeros(len(y)-1,dtype="int32"),p,1)^y


def dec_rev(a,n,m,y,k):
    r=remainder(a,n,m,y,k)
    if r==0:
        return np.array(y)
    else:
        p=position(r,n,m,y,k)
        if p==None:
            return 'failure'
        else:
            return Reversed(p,y)

def dec_alg1(a,n,m,y,k):
    if len(y)==n:
        return dec_sub(a,n,m,y,k)
    elif len(y)==n-1:
        return dec_del(a,n,m,y,k)
    else:
        return 'failure'

