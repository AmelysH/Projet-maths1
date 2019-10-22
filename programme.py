import autograd
from autograd import numpy as np
import matplotlib.pyplot as plt


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


def f(x,y):
    return np.sin(x)+2.0*np.sin(y)


def simple_contour(f,c=0.0,delta=0.01):
    
    def grad_f(x,y):
        gr=autograd.grad
        return np.r_[gr(f,0)(x,y),gr(f,1)(x,y)]

    def g(t):
        return f(0,t)

    x=[0.0]
    y=[float(find_seed(g,c))]

    i=0

    while abs(f(x[i],y[i])-c)<abs(c/100):
        grad=grad_f(x[i],y[i])
        aux=[grad[1],-grad[0]]
        norme_aux=np.sqrt(aux[0]**2+aux[1]**2)
        vect_orthon=[delta/norme_aux*aux[0],delta/norme_aux*aux[1]]
        posx=x[i]+vect_orthon[0]
        posy=y[i]+vect_orthon[1]
        if posx>1 or posx<0 or posy>1 or posy<0:
            posx=posx-2*vect_orthon[0]
            posy=posy-2*vect_orthon[1]
        elif posx>1 or posx<0 or posy>1 or posy<0:
            return x,y
        x.append(posx)
        y.append(posy)
        i+=1
    return x,y

def f_test(x,y):
    return np.power(x,2)+np.power(y,2)

x,y=simple_contour(f_test,0.25)
plt.plot(x,y)
plt.show()            




