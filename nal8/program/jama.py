import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
PATH = "../latex/pdfs/"
# Parameters
a = 5                # Width of the well
V0 = 1              # Well depth
N = 1000               # Number of points
x = np.linspace(-2*a, 2*a, N)
dx = x[1] - x[0]
# ticks = np.linspace(-a/2, a/2, 3)
ticks = [-2*a, -a, -a/2, 0, a/2, a, 2*a]
tick_labels = ["$-2a$","$-a$",r"$-\frac{a}{2}$", r"$0$",  r"$\frac{a}{2}$",'$a$', '$2a$']

# Define potential
V = np.zeros(N)
V[x > a/2] = V0
V[x < -a/2] = V0

# Construct Hamiltonian matrix
H = np.zeros((N, N))
for i in range(1, N-1):
    H[i, i] = 2 / dx**2 + V[i]
    H[i, i-1] = -1 / dx**2
    H[i, i+1] = -1 / dx**2

# Solve eigenvalue problem
E, psi = eigh(H)

# Normalize wave functions
psi /= np.sqrt(np.sum(psi**2, axis=0) * dx)

# Extract c1 and c2 for the first eigenstate
c1 = psi[int(N/2), 0]  # Wave function at x = a/2
c2 = (psi[int(N/2)+1, 0] - psi[int(N/2)-1, 0]) / (2 * dx)  # Finite difference for derivative


for i in range(4):

    En = E[i+2]
    A = 1/(np.sqrt(np.sum(psi[:, i+2]**2)*dx))
    l = plt.plot(x, A*psi[:, i+2]/2+En)
    col = l[0].get_color()

    plt.axhline(En, xmin=0, xmax=1, ls="--", c=col)
    if i >= 2:
        plt.text(-a*2+0.2, En - 0.06, f"$E_{i} = {En:.2f}$", color=col, fontsize=11)

    else:   
        plt.text(-a*2+0.2, En + 0.01, f"$E_{i} = {En:.2f}$", color=col, fontsize=11)

plt.plot(x, V, color="black", label="$V(x)$")
plt.legend()
plt.xticks(ticks, tick_labels)  # Set custom ticks and labels
plt.grid()
plt.title("Rešitve Schrödingerjeve enačbe za končno jamo z diferenčno metodo")
plt.xlabel("$x$")
plt.ylabel("$\psi_n(x)$")
plt.savefig(PATH + "jama.pdf")

