import autograd
from autograd import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def find_seed(g,c=0,debut=0,fin=1,eps=2**(-26)):
    if g(debut)<=c<=g(fin):
        a=debut
        b=fin
    elif g(fin)<=c<=g(debut):
        a=fin
        b=debut
    else :
        return None
    while b-a>eps:
        milieu=(a+b)/2
        if c<=g(milieu):
            b=milieu
        else :
            a=milieu
    return float((a+b)/2)


def f(x,y):
    return np.sin(x)+2.0*np.sin(y)



def simple_contour(f,c=0.0,delta=0.01):

    def grad_f(x,y):
        gr=autograd.grad
        return np.r_[gr(f,0)(x,y),gr(f,1)(x,y)]

    def g(t):
        return f(0,t)

    x=[0.0]
    y=[find_seed(g,c)]



    for i in range(1000):
        if x[-1]==None or y[-1]==None:
            return x[:-1],y[:-1]
        grad=grad_f(x[-1],y[-1])
        aux=[grad[1],-grad[0]]
        norme_aux=np.sqrt(aux[0]**2+aux[1]**2)
        vect_orthon=[delta/norme_aux*aux[0],delta/norme_aux*aux[1]]
        posx=x[-1]+vect_orthon[0]
        posy=y[-1]+vect_orthon[1]
        if posx+delta>1 or posx-delta<0 or posy+delta>1 or posy-delta<0:
            posx=posx-2*vect_orthon[0]
            posy=posy-2*vect_orthon[1]
            if posx+delta>1 or posx-delta<0 or posy+delta>1 or posy-delta<0:
                return x,y

        if grad[0]==0.0:
            x.append(posx)
            y.append(find_seed(lambda t:f(posx,t),c,posy-delta,posy+delta))

        else :
            t=find_seed(lambda t:f(t,posy+(t-posx)*grad[1]/grad[0]),c,posx-delta,posx+delta)
            x.append(t)
            try :
                y.append(posy+(t-posx)*grad[1]/grad[0])
            except:
                y.append(0)
    return x,y

def f_test(x,y):
    return np.power(x,2)+np.power(y,2)

plt.close()
x,y=simple_contour(f_test,0.5)
plt.plot(x,y)
plt.show()
