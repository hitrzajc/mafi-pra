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
Nt = 300
Nx = 300
# times = np.zeros((Nt, Nx), dtype=complex)
times = []
from time import time

for t_siz in tqdm(range(5,Nt,5)):
    times.append([])
    for x_siz in range(5,Nx,5):
        start = time()
        x = np.linspace(-40,40,x_siz)
        dx = x[1] - x[0]
        t = np.linspace(0, 10, t_siz)
        dt = t[1] - t[0]
        ψ_0 = ψ(x,0) #initial state

        sol = np.zeros((len(t), len(x)), dtype=complex)

        sol[0] = ψ_0

        for i in range(1,len(sol)):
            b = 1j*dt/2/(dx**2)
            a = -b/2
            d = 1j*V(x)*0.5*dt + b + 1


            A = np.diag(d) + np.diag(a*np.ones(len(x)-1),1) + np.diag(a*np.ones(len(x)-1),-1)
            vec = np.dot(np.conjugate(A), sol[i-1])
            # print(sol[i-1])
            sol[i] = LA.solve(A, vec)
        end = time()
        times[-1].append(end - start)

    

plt.imshow(times[::-1], cmap="hot",extent=[5, Nx, 5, Nt,], norm=LogNorm())
plt.colorbar(label="Čas [s]")
plt.xlabel("Število točk v x")
plt.ylabel("Število točk v t")
plt.title("Časovna zahtevnost")
plt.savefig(PATH + "time_log.pdf")
plt.clf()

plt.imshow(times[::-1], cmap="hot",extent=[5, Nx, 5, Nt,],)
plt.colorbar(label="Čas [s]")
plt.xlabel("Število točk v x")
plt.ylabel("Število točk v t")
plt.title("Časovna zahtevnost")
plt.savefig(PATH + "time.pdf")

