import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import scipy.integrate
quad = scipy.integrate.quad
#pde for diffusion equation 1D

PATH = "../latex/pdfs/"
N = 1000
x = np.linspace(0, 1, N)
dx = x[1]-x[0]
a = x[-1] - x[0]
t = np.linspace(0, 10, N)

a = x[-1] - x[0]
ticks = np.linspace(0,a, 3)
ticks_labels = ["$0$", r"$\frac{a}{2}$", "$a$"]
# T0 = np.sin(np.pi * 10 *x)
T0 = lambda x: np.exp(-10*(x-0.5)**2) if 0<x<a else 0

dt = t[1]-t[0]
D = 0.005

def scalar(f1,f2):
    tmp = lambda z: f1(z)*f2(z)
    return quad(tmp, 0, a)[0]

N_eig = 100
T = np.zeros((N,N))
T[0] = 0
T[-1] =0 

c0 = np.zeros(N_eig)
for n in range(N_eig):
    eigf = lambda z: np.sin(n*np.pi/a * z)
    c0[n] = scalar(eigf, T0) * 2/a
    for j in range(len(x)):
        T[0][j] += c0[n]*eigf(x[j])

# plt.plot(x,T[0])
# plt.show()
# T[0] = c0 * 
# print(c0[:10])
for i in range(1,len(t)):
    for n in range(N_eig):
        eigf = lambda z: np.sin(n*np.pi/a * z)

        T[i] += c0[n]*np.exp(-D*(n*np.pi/a)**2*(dt*i))*eigf(x)


    # for i in range(1,len(t)):




# exit()
# T = T.imag
cmap = cm.get_cmap("viridis",11)
img = plt.imshow(T, extent=[0, 1, 0, 1], aspect='auto', origin='lower',cmap = cmap)
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar(img, label="Temperatura")
plt.title("Heat map - dirichlet")
plt.xticks(ticks, ticks_labels)
plt.savefig(PATH + "heat_map_direchlet.pdf")
plt.clf()
plt.xlabel("x")
plt.ylabel("Temperatura")
# cm.vulc
N = 200
colors = cm.inferno(np.linspace(0, 1, 10))
for i in range(N):
    idx = int(len(t)//N * i)
    cidx = int(len(colors)/N *i)
    time = (t[-1] - t[0])/(N)*i
    plt.plot(x,T[idx],color=colors[len(colors)-cidx-1])

sm = plt.cm.ScalarMappable(cmap=cm.inferno.reversed(), norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])  # Needed for ScalarMappable without image data
plt.colorbar(sm, boundaries=np.linspace(0, 1, 10), label="čas")
plt.xlim(0,1)
# plt.ylim(T.min(),T.max())
plt.title("Heat map - dirichlet")

plt.xticks(ticks, ticks_labels)
# plt.
# plt.show()
plt.savefig(PATH + "heat_map2_direchlet.pdf")


def init():
    line.set_data([], [])
    return line,

# Update function
def update(frame):
    y = T[frame]
    line.set_data(x, y)
    return line,
plt.clf()

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 1)
ax.set_ylim(T.min(), T.max())
ax.set_xlabel("Position")
ax.set_ylabel("Value")
ax.set_title("Time vs Position")

# # Create animation
# ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)

# # Save as a GIF
# ani.save("animation.gif", writer=PillowWriter(fps=10))