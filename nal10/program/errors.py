import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm
from tqdm import tqdm

PATH = "../latex/pdf/"
# print(plt.rcParams.keys())
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.pad_inches'] = 0
ω = 0.2
λ = 10

def ψ(x, t, ω=0.2, λ=10): #analiticna resitu
    α = np.sqrt(ω)
    ξ_λ = α * λ
    ξ = α * x

    expo = -0.5 * (ξ - ξ_λ*np.cos(ω*t))**2 - 1j * (0.5 * ω*t + ξ*ξ_λ*np.sin(ω*t) - 0.25*ξ_λ**2*np.sin(2*ω*t))
    return np.exp(expo) * np.sqrt(α/np.sqrt(np.pi))

def V(x, k = np.sqrt(ω)):
    return 0.5*x**2*k


x = np.linspace(-40,40,300)
dx = x[1] - x[0]
t = np.linspace(0, 10, 300)
dt = t[1] - t[0]
ψ_0 = ψ(x,0) #initial state

sol = np.zeros((len(t), len(x)), dtype=complex)

sol[0] = ψ_0


# A = np.array([[1, 2], [3, 4]])  # 2x2 matrix
# sol = np.array([1, 2])  # 1D vector
# print(np.dot(A, sol),"here")  # 1D vector
# plt.plot(x, sol[0].real)


for i in range(1,len(sol)):
    b = 1j*dt/2/(dx**2)
    a = -b/2
    d = 1j*V(x)*0.5*dt + b + 1

    
    A = np.diag(d) + np.diag(a*np.ones(len(x)-1),1) + np.diag(a*np.ones(len(x)-1),-1)
    vec = np.dot(np.conjugate(A), sol[i-1])
    # print(sol[i-1])
    sol[i] = LA.solve(A, vec)



analytic = np.zeros((len(t), len(x)), dtype=complex)
for i in range(0,len(t)):
    t1 = t[i]
    analytic[i] = ψ(x,t1)

error = np.abs(np.abs(analytic) - np.abs(sol))
plt.imshow(error, aspect="auto", extent=[x.min(), x.max(), t.min(), t.max()], cmap='viridis')
plt.colorbar(label='Absoulte error')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Error between Analytic and Numerical Solutions')
plt.savefig(PATH+"abs_error.png")
plt.clf()

plt.imshow(error, aspect="auto", extent=[x.min(), x.max(), t.min(), t.max()], cmap='viridis', norm=LogNorm())
plt.colorbar(label='Absoulte error (log scale)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Error between Analytic and Numerical Solutions (Log Scale)')
plt.savefig(PATH+"abs_error_log.png")
# plt.show()




