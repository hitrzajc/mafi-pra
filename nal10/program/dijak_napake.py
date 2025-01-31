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

xx = []
yy = []
for N in tqdm(range(2, 22, 2)):
    c = FD(2,N, N//2)
    
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

    analytic = np.zeros((len(t), len(x)), dtype=complex)
    for i in range(0,len(t)):
        t1 = t[i]
        analytic[i] = ψ(x,t1)

    error = np.abs(np.abs(analytic) - np.abs(sol))
    mx = -1
    for i in range(len(error)):
        for j in range(len(error[i])):
            mx = max(mx, error[i][j])
    yy.append(mx)
    xx.append(N)

# plt.grid()
x = xx
y = np.log(yy)

lower_limit = 10
upper_limit = 165

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Plot data in both subplots
# plt.grid()
ax1.scatter(x, y,color="black")
ax2.scatter(x, y,color="black")

# Set y-limits for each subplot
ax1.set_ylim(upper_limit, max(y)+1)  # Top part
ax2.set_ylim(min(y), lower_limit)  # Bottom part

# Remove spines between subplots
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Add diagonal lines to indicate break
d = .015  # Size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)  # Top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

# Adjust layout
plt.subplots_adjust(hspace=0.05)
# plt.show()# plt.yscale("log")
# plt.ylim(bottom=0, top=np.log(yy).max() + 1)
plt.xlabel("N")
plt.ylabel("$\log$ Max error")
# plt.title("Max error vs Število členov v odvodu")
plt.xticks(np.arange(0, 21, 2))
plt.savefig(PATH+"max_error.pdf")