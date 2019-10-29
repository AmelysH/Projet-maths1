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
    while abs(g((a+b)/2)-c)>eps:
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

    i=0
    
    while True:
        if x[-1]==None or y[-1]==None:
            return x[:-1],y[:-1]
        grad=grad_f(x[-1],y[-1])
        aux=[grad[1],-grad[0]]
        norme_aux=np.sqrt(aux[0]**2+aux[1]**2)
        vect_orthon=[delta/norme_aux*aux[0],delta/norme_aux*aux[1]]
        posx=x[-1]+vect_orthon[0]
        posy=y[-1]+vect_orthon[1]

        if i==0:
            if posx>1 or posx<0 or posy>1 or posy<0:
                posx=posx-2*vect_orthon[0]
                posy=posy-2*vect_orthon[1]
                if posx>1 or posx<0 or posy>1 or posy<0:
                    return x,y

        elif i>=1 :
            if vect_orthon[0]*(x[i]-x[i-1])+vect_orthon[1]*(y[i]-y[i-1])<0:
                posx=posx-2*vect_orthon[0]
                posy=posy-2*vect_orthon[1]
            if posx>1 or posx<0 or posy>1 or posy<0:
                return x,y

        i+=1

        if grad[0]==0.0:
            x.append(posx)
            y.append(find_seed(lambda t:f(posx,t),c,max(posy-delta,0),min(posy+delta,1)))

        else :
            t=find_seed(lambda t:f(t,posy+(t-posx)*grad[1]/grad[0]),c,max(posx-delta,0),min(posx+delta,1))
            x.append(t)
            try :
                y.append(posy+(t-posx)*grad[1]/grad[0])
            except:
                y.append(0)
    return x,y

def f_test(x,y):
    return np.power(x,2)+np.power(y-1,2)

def g(x,y):
    return np.exp(-np.power(x,2)-np.power(y,2))
def h(x,y):
    return g(x-1,y-1)

def f_test2(x,y):
    return 2*(g(x,y)-h(x,y))



def rotation(g):      #Si g est définie sur [0,1]x[0,1], renvoie la fonction g rotationnée de pi/2
    return lambda x,y : g(y,1-x)

def contour(f,c=0.0,xc=[0.0,1.0], yc=[0.0,1.0], delta=0.01):

    xs=[]
    ys=[]
    for i in range(len(xc)-1):
        for j in range(len(yc)-1):
            deb_x=xc[i]
            fin_x=xc[i+1]
            deb_y=yc[j]
            fin_y=yc[j+1]
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

            ajout_renormalisation(x_gauche,y_gauche)
            ajout_renormalisation(y_haut,[1-x for x in x_haut])
            ajout_renormalisation([1-x for x in x_droite],[1-y for y in y_droite])
            ajout_renormalisation([1-y for y in y_bas],x_bas)

    return xs,ys


from scipy import*
quadri=linspace(-2,3,10).tolist()
plt.close()
for n in [-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5]:
    x,y=contour(f_test2,n,quadri,quadri)
    plt.scatter(x,y,color='blue',s=1)
plt.xlim(-2.0,3.0)
plt.ylim(-1.0,2.0)
plt.show()



