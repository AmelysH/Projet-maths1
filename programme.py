import autograd
from autograd import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def find_seed(g,c=0,debut=0,fin=1,eps=2**(-26)):
    if g(debut)<=c<=g(fin) or g(fin)<=c<=g(debut):
        a=debut
        b=fin
    else :
        return None
    while b-a>eps:
        milieu=(a+b)/2
        if g(a)<=c<=g(milieu) or g(a)>=c>=g(milieu):
            b=milieu
        else :
            a=milieu
    return float((a+b)/2)


def f(x,y):
    return np.sin(x)+2.0*np.sin(y)


def simple_contour(f,c=0.0,delta=0.01):

    def grad_f(x,y):
        gr=autograd.grad
        return [gr(f,0)(x,y),gr(f,1)(x,y)]

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
    return np.power(x,2)+np.power(y-1,2)



def rotation(g):      #Si g est définie sur [0,1]x[0,1], renvoie la fonction g rotationnée de pi/2
    return lambda x,y : g(y,1-x)

def contour(f,c=0.0,xc=[0.0,1.0], yc=[0.0,1.0], delta=0.01):

    xs=[]
    ys=[]
    for i in range(len(xc)-1):
        deb_x=xc[i]
        fin_x=xc[i+1]
        deb_y=yc[i]
        fin_y=yc[i+1]
        ecart_x=fin_x-deb_x
        ecart_y=fin_y-deb_y

        def f_nor(x,y):                # On se ramène à une fonction 'normalisée' définie sur [0,1]x[0,1]
            return f(x*ecart_x+deb_x,y*ecart_y+deb_y)

        def ajout_renormalisation(liste_x,liste_y):
            for x in liste_x:
                xs.append(x*ecart_x+deb_x)
            for y in liste_y:
                ys.append(y*ecart_y+deb_y)

        x_gauche,y_gauche=simple_contour(f_nor,c,delta)
        x_haut,y_haut=simple_contour(rotation(f_nor),c,delta)
        x_droite,y_droite=simple_contour(rotation(rotation(f_nor)),c,delta)
        x_bas,y_bas=simple_contour(rotation(rotation(rotation(f_nor))),c,delta)

        print(x_droite,y_droite)
        print(x_haut[0],y_haut[0])
        print(x_gauche[0],y_gauche[0])
        print(x_bas,y_bas)

        ajout_renormalisation(x_gauche,y_gauche)
        ajout_renormalisation(x_haut,y_haut)
        ajout_renormalisation(x_droite,y_droite)
        ajout_renormalisation(x_bas,y_bas)
        #ajout_renormalisation([1-y for y in y_haut],x_haut)
        #ajout_renormalisation([1-x for x in x_droite],[1-y for y in y_droite])
        #ajout_renormalisation([y for y in y_bas],[1-x for x in x_bas])

    return xs,ys

plt.close()
x,y=contour(f_test,0.5)
plt.plot(x,y)
plt.axis('equal')
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)
plt.show()


