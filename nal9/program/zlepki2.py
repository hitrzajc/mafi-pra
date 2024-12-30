import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
#pde for diffusion equation 1D
from scipy.linalg import solve_banded
from scipy.linalg import inv

plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
PATH = "../latex/pdfs/"
N = 500
a=1
x = np.linspace(0, a, N)
dx = x[1]-x[0]
t = np.linspace(0, a, N)
dt = t[1]-t[0]
D = 0.005

N_splits = 20
xk = np.linspace(0, a, N_splits)
def BB(k, _x):
    dx = xk[1]-xk[0]
    if(k<2) or k>N_splits-3:
        return 0
    # Assuming x is a list [x_{k-2}, x_{k-1}, x_k, x_{k+1}, x_{k+2}]
    if _x <= xk[k-2]:  # x <= x_{k-2}
        return 0
    elif xk[k-2] <= _x <= xk[k-1]:  # x_{k-2} <= x <= x_{k-1}
        return (1 / dx**3) * (_x - xk[k-2])**3
    
    elif xk[k-1] <= _x <= xk[k]:  # x_{k-1} <= x <= x_k
        return ((1 / dx**3) * (_x - xk[k-2])**3 -
                (4 / dx**3) * (_x - xk[k-1])**3)
    
    elif xk[k] <= _x <= xk[k+1]:  # x_k <= x <= x_{k+1}
        return ((1 / dx**3) * (xk[k+2] - _x)**3 -
                (4 / dx**3) * (xk[k+1] - _x)**3)
    
    elif xk[k+1] <= _x <= xk[k+2]:  # x_{k+1} <= x <= x_{k+2}
        return (1 / dx**3) * (xk[k+2] - _x)**3
    
    else:
        return 0


# T0 = np.sin(np.pi * 10 *x)
T0 = np.sin(np.pi*(xk)**2)
T0[0] = T0[-1] = 0

main_diag = 4 * np.ones(N_splits)       # Main diagonal with 4s
off_diag = 1 * np.ones(N_splits - 1)    # Off-diagonals with 1s
ab = np.zeros((3, N_splits))
ab[0, 1:] = off_diag   # Upper diagonal
ab[1, :] = main_diag   # Main diagonal
ab[2, :-1] = off_diag  # Lower diagonal
# Construct the tridiagonal matrix
A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
c = solve_banded((1, 1), ab, T0)

main_diag = -2 * np.ones(N_splits)       # Main diagonal with 4s
off_diag = 1 * np.ones(N_splits - 1)  
B = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
print(A[:5,:5])
print(B[:5,:5])
Ai = inv(A)
factor = 6*D/((xk[1]-xk[0])**2)
T = np.zeros((len(t),len(x)))


for i in range(0, len(t)):
    for k in range(0,N_splits):
        for j in  range(len(x)):
            T[i][j] += BB(k, x[j])*c[k]
            # print(BB(k, x[j])*c[k])
    # break
    # print(i)
    c = c + dt*Ai @ (B @ c)*factor
    # plt.plot(x,T[i])
    # plt.show()


cmap = cm.get_cmap("viridis",11)
img = plt.imshow(T, extent=[0, 1, 0, 1], aspect='auto', origin='lower',cmap = cmap)
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar(img, label="Temperatura")
plt.title("Heat map - dirichlet-zlepki")# y = np.zeros(N)
plt.savefig(PATH+"zlepki2.pdf")
# plt.show()