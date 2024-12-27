import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib import cm

PATH =  "../latex/pdfs/"
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
def draw_potential(a):
    x = np.linspace(-a/2, a/2, 1000)
    y = np.zeros(1000)
    plt.plot(x, y, color="black")
    y_min, y_max = plt.gca().get_ylim()  
    plt.plot([-a/2, -a/2], [0, y_max], color='black', label="$V(x)$")
    plt.plot([a/2, a/2], [0, y_max], color='black')
    plt.ylim(y_min, y_max)

def V(x):
    V0 = 1
    if x < -a/2 or x > a/2:
        return V0
    else:
        return 0
    
def f(x,y,E):
    y1, y2 = y #y1 = y, y2 = y'
    return [y2, -y1*(E-V(x))]

def find(f, y0, enrgija, x_span,y1): #funckija zacetni/robni pogoji
    E0,E1 = enrgija
    i=0
    t_span=[x_span[0],x_span[-1]]
    while E1-E0 > 1e-8:
        E = (E0 + E1)/2
        sol = solve_ivp(f,t_span=t_span,t_eval=x_span, y0=y0, args=(E,), method='RK45')
        # E1-E0

        if sol.y[0, -1] > y1[0]: #abusing simetry
            E0 = E
        else:
            E1 = E
        # i+=1
        # print(i)
        
        # plt.plot(x_span, sol.y[0], color=colors[i-1],linewidth=2)

        # if i == N:break
    return (E1+E0)/2, sol
    # sol = solve_ivp(f, interval, [y0,y1], args=(E0,), method='RK45')
a = 5
x_span = np.linspace(-a*2,a*2,10000)

y0 = [0,0.01]
E = 0.5


