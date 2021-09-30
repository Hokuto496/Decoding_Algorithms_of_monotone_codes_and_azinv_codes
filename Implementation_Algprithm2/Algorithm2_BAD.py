import numpy as np

"""
Here n is the codeword length and y is the noisy codeword.
In addition, a and m are the parameters of the azinv code.
We determine the insertion position and the alphabet to be inserted 
according to the result of the comparison of 
the value of the function remainder and the value of the function weight.
"""

#Compute how much it differs from the original codeword

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


#Compute a basis for determining how much it differs from the original codeword

def evenflipped(y):
    return y^np.arange(len(y))%2

def weight(y,k=0):
    a=evenflipped(y)
    return np.sum(a)

#Insertion position 1
def position1(y,r,k=0):
    arr=np.array(evenflipped(y))
    pos=np.count_nonzero(arr==1)
    if pos==r:
            return len(arr)
    for i in np.arange(len(arr))[::-1]:
        pos-=arr[i]
        if pos==r:
            return i

#Insertion position 2

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

#Insertion alphabets 1
def deleted_seq1(p):
    if p%2==0:
        return [0,1]
    else:
        return [1,0]

#Insertion alphabets 2
def deleted_seq2(p):
    if p%2==0:
        return [1,0]
    else:
        return [0,1]

#Function to insert b at the p-th position of the series y
def ins(y,p,b):
    return np.insert(y,p,b)

#Decode for BAD
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
Length of the code word n=5, integers a=0 and m=5, and series y=011.
Check if y can be decoded into the element 01011 of the azinv code
A_{a,m}(n)={01000,01010,01011,01111,10110,10001} in this case.
"""

#011011
dec_BAD(0,5,5,[0,1,1])
