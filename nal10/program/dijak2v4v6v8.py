import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm
from tqdm import tqdm
import sympy as sp

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

def FD(m, n, s):
    x = sp.Symbol('x')
    expr = (x**s * sp.log(x)**m)
    series_exp = sp.series(expr, x, x0=1, n=n+1).removeO()

    dic = sp.collect(sp.expand(series_exp), x).as_coefficients_dict()
    arr = []
    for key in dic:
        base,exp = key.as_base_exp()
        arr.append((exp,dic[key]))
    arr.sort()
    out = []
    for i in arr:
        out.append(float(i[1]))
    return out




# for i in t:
#     print(i,t[i])

x = np.linspace(-40,40,300)
dx = x[1] - x[0]
t = np.linspace(0, 10, 300)
dt = t[1] - t[0]
ψ_0 = ψ(x,0) #initial state

sol = np.zeros((len(t), len(x)), dtype=complex)
sol[0] = ψ_0

N = 2
c = FD(2,N,2)
# print(c)
d = (1+1j*dt/2*V(x))
A = np.diag(d)

cof = -1j*dt/4/(dx**2)
for r in range(len(c)):
    pos = r-N//2
    A += cof*np.diag(c[r]*np.ones(len(x)-abs(pos)),pos)

Ac = np.conjugate(A)

for i in range(1,len(sol)):  
    vec = np.dot(Ac, sol[i-1])
    sol[i] = LA.solve(A, vec)


N = 4
c = FD(2,N,N//2)
d = (1+1j*dt/2*V(x))
A = np.diag(d)

cof = -1j*dt/4/(dx**2)
for r in range(len(c)):
    pos = r-N//2
    A += cof*np.diag(c[r]*np.ones(len(x)-abs(pos)),pos)

Ac = np.conjugate(A)

analytic = np.zeros((len(t), len(x)), dtype=complex)
for i in range(1,len(sol)):  
    vec = np.dot(Ac, sol[i-1])
    analytic[i] = LA.solve(A, vec)

error = np.abs(np.abs(analytic) - np.abs(sol))
plt.imshow(error[::-1], aspect="auto", extent=[x.min(), x.max(), t.min(), t.max()], cmap='viridis',norm=LogNorm())
plt.colorbar(label='Absoulte error')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Diference between Numerical Solutions N=2, N=4'.format(N))
plt.savefig(PATH+"abs_error_log_2_4.pdf")
# plt.show()
plt.clf()