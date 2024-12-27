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

def f(x,y,E):
    y1, y2 = y #y1 = y, y2 = y'
    return [y2, -y1*E]

def find(f, y0, enrgija, x_span,y1): #funckija zacetni/robni pogoji
    E0,E1 = enrgija
    i=0
    t_span=[x_span[0],x_span[-1]]
    while E1-E0 > 1e-8:
        E = (E0 + E1)/2
        sol = solve_ivp(f,t_span=t_span,t_eval=x_span, y0=y0, args=(E,), method='RK45')
        # E1-E0

        if sol.y[0, -1] > y1[0]:
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
x_span = np.linspace(-a/2,a/2,10000)



h = x_span[1]-x_span[0]

E,sol = find(f, [0,2], [0, 1], x_span,[0,1])
A = 1 / (np.sqrt(sum(sol.y[0]**2)*h))
E = E * A
print(A)
sol.y[0] = sol.y[0] * A

print(E)
plt.plot(x_span, sol.y[0], label="$E = {:.3f}$".format(E),linewidth=2)
plt.title("Rešitve Schrödingerjeve enačbe za neskončno jamo s strelsko metodo")

plt.legend()
ticks = np.linspace(-a/2, a/2, 5)
tick_labels = [r"$-\frac{a}{2}$", r"$-\frac{a}{4}$", r"$0$", r"$\frac{a}{4}$", r"$\frac{a}{2}$"]
plt.xticks(ticks, tick_labels)  # Set custom ticks and labels
plt.grid()
plt.xlabel("x")
plt.ylabel(r"$\psi(x)$")
draw_potential(a)
# plt.plot(x_span, sol.y[1])

plt.savefig(PATH+"neskoncna_jama_strelska.pdf")
# plt.show()
plt.clf()


# plt.show()
N = 20
colors = cm.plasma(np.linspace(0, 1, N))
# colors = cm.tab20b(np.linspace(0, 1, N))
print(len(colors))
Es = np.linspace(0,0.5,N)

for i in range(N):
    t_span=[x_span[0],x_span[-1]]

    sol = solve_ivp(f,t_span=t_span,t_eval=x_span, y0=[0,1], args=(Es[i],), method='RK45')

    A = 1 / (np.sqrt(sum(sol.y[0]**2)*h))
    AA = A
    plt.plot(x_span, sol.y[0]*A, color=colors[i],linewidth=3,solid_capstyle='butt')
plt.title("Rešitve Schrödingerjeve enačbe z različnimi fiksnimi energijami")
plt.xticks(ticks, tick_labels)  # Set custom ticks and labels
# plt.grid()
plt.xlabel("x")
plt.ylabel(r"$\psi(x)$")
plt.autoscale()

draw_potential(a)


E,sol = find(f, [0,2], [0, 1], x_span,[0,1])
A = 1 / (np.sqrt(sum(sol.y[0]**2)*h))
plt.plot(x_span,sol.y[0]*A,color='black',linestyle='dashed',label="Prava rešitev")

import matplotlib as mpl
import matplotlib.lines as mlines

cbar = plt.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(0,AA*0.5),cmap=cm.plasma),label="Energija")

# print(A,AA,colors[-5])
y_position = cbar.ax.transAxes.transform((0, 1))[1]
y_position = 0.7
print(y_position,A*E)
line = mlines.Line2D([0.1, 0.9], [y_position, y_position], color='black', linestyle='--', linewidth=0.7, transform=cbar.ax.transAxes)
cbar.ax.add_line(line)

plt.legend()

plt.savefig(PATH+"neskoncna_jama_strelska_vec.pdf")

plt.clf()