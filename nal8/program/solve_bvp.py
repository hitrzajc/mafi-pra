import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar

np.seterr(invalid="ignore")
a = 5
def V(x, V0=50, a=5):
    if np.iterable(x):
        return np.array([V(xi, V0, a) for xi in x])
    elif np.abs(x) < a/2:
        return 0
    else:
        return V0
    
def fun(x, y, p):
    E = p[0]
    return np.vstack((y[1], -2*(E - V(x))*y[0]))

def bc(ya, yb, p):
    return np.array([ya[0], yb[0], ya[1] - 0.001])

N = 1000
x = np.linspace(-3, 3, N)
y_i = np.zeros((2, x.size))
y_i[0,4] = 0.1



def fk(E, V0=50, a=5, n=1):
    """Returns the error in the equality:
        k2 = k1 * tan(k1 * a / 2) for odd n (even parity solutions); or
       -k2 = k1 * cot(k1 * a / 2) for even n (odd parity solutions),
        where k1 = sqrt(2 * E) and k2 = sqrt(2 * (V0 - E)) for E < V0.
    """
    k1 = np.sqrt(2 * E)
    k2 = np.sqrt(2 * (V0 - E))
    if n % 2:
        return k2 - k1 * np.tan(k1 * a / 2)
    else:
        return k2 + k1 / np.tan(k1 * a / 2)


def Eanalytic(V0=50, a=5, pts=N):
    """Finds the roots of the fk between 0 and V0 for odd and even n."""
    Ei = np.linspace(0.0, V0, pts)
    roots = []
    for n in [1, 2]:
        for i in range(pts - 1):
            soln = root_scalar(fk, args=(V0, a, n), x0=Ei[i], x1=Ei[i + 1])
            if soln.converged and np.around(soln.root, 9) not in roots:
                roots.append(np.around(soln.root, 9))
    return np.sort(roots)


Elist = Eanalytic()
print(Elist)

solns = [solve_bvp(fun, bc, x, y_i, p=[Ei]) for Ei in Elist]
x_plot = np.linspace(x.min(), x.max(), N)
plt.plot(x_plot, V(x_plot), drawstyle='steps-mid', c='k', alpha=0.5)
for soln in solns[:3]:
    y_plot = soln.sol(x_plot)[0]
    l = plt.plot(x_plot, 4 * y_plot / y_plot.max() + soln.p[0])
    plt.axhline(soln.p[0], xmin=0.25, xmax=0.75, ls='--', c=l[0].get_color())
plt.axis(xmin=-3, xmax=3)
plt.xlabel(r'$x$')
plt.ylabel(r'$\psi(x)$')
plt.show()