import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

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

    # plt.plot(x, sol[i].real)


# plt.show()

#animate
# plt.xlim(x.min(), x.max())
# plt.ylim(sol.min(), sol.max())
# plt.plot(x,V(x),color="black", label="V(x) - potencial")
# plt.xlabel("x")
# plt.ylabel("ψ(x)")
# plt.legend()
# line, = plt.plot(x, sol[0], lw=2,color="red")
# def update(frame):
#     line.set_ydata(np.abs(sol[frame]))
#     return line,
# ani = FuncAnimation(plt.gcf(), update, frames=len(t), blit=True)
# writer = FFMpegWriter(fps=20, bitrate=1800)
# ani.save("animation.mp4", writer=writer, dpi=150)
# plt.close()
# Create figures for real, imaginary, and absolute parts of the solution

def animate_propagation(sol, name):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.tight_layout()

    # Real part
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(sol.real.min(), sol.real.max())
    ax1.plot(x, V(x), color="black", label="V(x) - potential")
    ax1.set_xlabel("x")
    # ax1.set_ylabel("Re(ψ(x))")
    line1, = ax1.plot(x, sol[0].real, lw=2, color="red",label="$\Re\psi(x)$")
    ax1.legend()

    # Imaginary part
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(sol.imag.min(), sol.imag.max())
    ax2.plot(x, V(x), color="black", label="V(x) - potential")
    ax2.set_xlabel("x")
    # ax2.set_ylabel("Im(ψ(x))")
    line2, = ax2.plot(x, sol[0].imag, lw=2, color="blue",label="$\Im\psi(x)$")
    ax2.legend()

    # Absolute value
    ax3.set_xlim(-15, 15)
    ax3.set_ylim(np.abs(sol).min(), np.abs(sol).max())
    ax3.plot(x, V(x), color="black", label="V(x) - potential")
    ax3.set_xlabel("x")
    # ax3.set_ylabel("|ψ(x)|")
    line3, = ax3.plot(x, np.abs(sol[0]), lw=2, color="green",label="$|\psi(x)|$")
    ax3.legend()

    def update_all(frame):
        line1.set_ydata(sol[frame].real)
        line2.set_ydata(sol[frame].imag)
        line3.set_ydata(np.abs(sol[frame]))
        return line1, line2, line3

    ani = FuncAnimation(fig, update_all, frames=len(t), blit=True)
    writer = FFMpegWriter(fps=20, bitrate=1800)
    print("[*] Saving animation "+name+".mp4")
    ani.save("../latex/mp4/"+name+".mp4", writer=writer, dpi=150)
    print("[*] Animation {}.mp4 saved".format(name))
    plt.close()
    plt.clf()

## acutual work
# animate_propagation(sol, "naive_aproach")

analytic = np.zeros((len(t), len(x)), dtype=complex)
for i in range(0,len(t)):
    t1 = t[i]
    analytic[i] = ψ(x,t1)

## actual work
# animate_propagation(analytic, "analytic_solution")

# ploting absolute error between analytic and numerical solution


