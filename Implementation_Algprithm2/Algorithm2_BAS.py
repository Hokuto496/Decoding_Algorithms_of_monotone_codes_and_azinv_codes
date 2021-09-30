import numpy as np

"""
Here n is the codeword length and y is the noisy codeword.
In addition, a and m are the parameters of the azinv code.
We determine the substitution position  
according to the value of the function remainder.
"""

#Compute how much it differs from the original codeword

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

#Substitution position
def position(r,n,m,y,k=0):
    return len(y)-min([r,m-r])

#Function to substitute for the p-th of the series y
def Substituted(p,y):
    return np.insert(np.zeros(len(y)-2,dtype="int32"),p-1,[1,1])^y

#Decode for BAS
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

"""
Length of the code word n=6, integers a=0 and m=10, and series y=011011.
Check if y can be decoded into the element 010111 of the azinv code
A_{a,m}(n)={011111,0101111,010101,010100,010000} in this case.
""" 
#[0, 1, 0, 1, 1, 1] 
dec_BAS(0,6,10,[0,1,1,0,1,1])
