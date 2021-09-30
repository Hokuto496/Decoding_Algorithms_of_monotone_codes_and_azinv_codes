import numpy as np

#Algortim 2 for BAD

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

def evenflipped(y):
    return y^np.arange(len(y))%2

def weight(y,k=0):
    a=evenflipped(y)
    return np.sum(a)

def remainder(a,n,m,y,k=0):
    return (a-inversion_number(azby_inverse(y)))%m


def flipped(b):
    if b==0:
        return 1
    elif b==1:
        return 0

def position1(y,r,k=0):
    arr=np.array(evenflipped(y))
    pos=np.count_nonzero(arr==1)
    if pos==r:
            return len(arr)
    for i in np.arange(len(arr))[::-1]:
        pos-=arr[i]
        if pos==r:
            return i

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

def deleted_seq1(p):
    if p%2==0:
        return [0,1]
    else:
        return [1,0]

def deleted_seq2(p):
    if p%2==0:
        return [1,0]
    else:
        return [0,1]

def I(y,p,b):
    return np.insert(y,p,b)

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

#Algorithm for BAS

def position(r,n,m,y,k=0):
    return len(y)-min([r,m-r])

def Substituted(p,y):
    return np.insert(np.zeros(len(y)-2,dtype="int32"),p-1,[1,1])^y

def dec_BAS(a,n,m,y,k=0):
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
            return Substituted(p,y)
        
#Algorithm 2
def dec_alg2(a,n,m,y,k=0):
    if len(y)==n:
        return dec_BAS(a,n,m,y,k)
    elif len(y)==n-2:
        return dec_BAD(a,n,m,y,k)
    else:
        return 'failure'
