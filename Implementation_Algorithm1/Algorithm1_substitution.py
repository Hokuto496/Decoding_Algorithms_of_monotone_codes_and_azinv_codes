import numpy as np

"""
Here n is the codeword length and y is the noisy codeword.
In addition, a, m, and k are the parameters of the monotone code.
We determine the substitution position  
according to the value of the function remainder.
"""

#Compute how much it differs from the original codeword
def remainder(a,n,m,y,k):
     return (a-np.dot(k[0:len(y)],y))%m

#Substitution position
def position(r,n,m,y,k):
    const=min([r,m-r])
    for i in np.arange(len(y)):
        if k[i]==const:
            return i
#Function to substitute for the p-th of the series y          
def Substituted(p,y):
     return np.insert(np.zeros(len(y)-1,dtype="int32"),p,1)^y

#Decode for substitution 
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

"""
Length of the code word n=6, integers a=1 and m=20, 
sequence k=(1,2,3,8,9,10), and series y=110000.
Check if y can be decoded into the element 100000 of the monotone code
M_{a,m,k}(n)={100000,110101,101110,010011,001101} in this case.
""" 
#[1,0,0,0,0,0]     
dec_sub(1,6,20,[1,1,0,0,0,0],[1,2,3,8,9,10])
