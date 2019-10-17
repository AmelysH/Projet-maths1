def find_seed(g,c=0,eps=2**(-26)):   
    if g(0)<=c<=g(1):
        a=0
        b=1
    elif g(1)<=c<=g(0):
        a=1
        b=0
    else :
        return None
    while b-a>eps:
        milieu=(a+b)/2
        if c<=g(milieu):
            b=milieu
        else :
            a=milieu
    return (a+b)/2

def f(x):
    return 2*x-1

#find_seed(f)

import autograd
from autograd import numpy as np

def grad_f(x,y):
    g=autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]


#def simple_contour(f,c=0.0,delta=0.01):


p