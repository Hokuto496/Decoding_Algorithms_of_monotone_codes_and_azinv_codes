import numpy as np

"""
Here n is the codeword length and y is the noisy codeword.
In addition, a, m, and k are the parameters of the monotone code.
We determine the insertion position and the alphabet to be inserted 
according to the result of the comparison of 
the value of the function remainder and the value of the function weight.
"""

#Compute how much it differs from the original codeword
def remainder(a,n,m,y,k):
    return (a-np.dot(k[:len(y)],y))%m 

#Compute a basis for determining how much it differs from the original codeword
def weight(y,k):
    return np.dot(np.diff(k,n=1)[:len(y)],y)

#Insertion position 1
def position1(r,y,k):
    pos=0
    if pos==r:
        return len(y)
    for i in np.arange(len(y))[::-1]:
        pos+=y[i]*(k[i+1]-k[i])
        if pos==r:
            return i
        
#Bit flip
def flipped(n):
    if n==0:
        return 1
    elif n==1:
        return 0

#Insertion position 2
def position2(r,y,k):
    pos=0
    const=r-weight(y,k)-k[0]
    if pos==const:
        return 0
    for i in np.arange(len(y)):
        pos+=flipped(y[i])*(k[i+1]-k[i])
        if pos==const:
            return i+1

#Insertion alphabet 1
def deleted_seq1(p):
    return 0

#Insertion alphabet 1
def deleted_seq2(p):
    return 1

#Function to insert b at the p-th position of the series y
def ins(p,b,y):
    return np.insert(y,p,b)

#Decode for deletion
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

"""
Length of the code word n=5, integers a=2 and m=9, 
sequence k=(1,2,3,6,8), and series y=0111.
Check if y can be decoded into the element 01110 of the monotone code
M_{a,m,k}(n)={01000,11001,01110,00101,11111} in this case.
"""
#01110
dec_del(2,5,9,[0,1,1,1],[1,2,3,6,8])
