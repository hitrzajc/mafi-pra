import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
#pde for diffusion equation 1D

plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
PATH = "../latex/pdfs/"
N = 1000
x = np.linspace(0, 1, N)
dx = x[1]-x[0]
t = np.linspace(0, 10, N)
# T0 = np.sin(np.pi * 10 *x)
T0 = np.sin(np.pi*(x)**2)
T0[0] = T0[-1] = 0


# Tk0 = fft.fft(T0,N)
# fk = fft.fftfreq(n=N,d=dx)

Tk = np.zeros((N,N),dtype=complex)
Tk0 = fft.fft(T0,N)

fk = fft.fftfreq(n=N,d=dx)
# fk = fft.fftshift(fk)
Tk[0] = Tk0
h = t[1]-t[0]
D = 0.005
T = np.zeros((N,N))
T[0] = T0
a = x[-1] - x[0]
ticks = np.linspace(0,a, 3)
ticks_labels = ["$0$", r"$\frac{a}{2}$", "$a$"]
# plt.plot(x,T[0])
# plt.plot(fft.fftshift(fk),fft.fftshift(abs(Tk[0])))
# plt.show()
for i in range(1,len(t)):
    for j in range(len(x)):
        Tk[i][j] = Tk[0][j] * np.exp(-4*np.pi**2*D*(fk[j])**2*t[i])
    T[i] = np.real(fft.ifft(Tk[i]))

# plt.plot(x,T[0])
# plt.plot(x,T[1])
# plt.plot(x,T[500])
# plt.show()

# exit()
cmap = cm.get_cmap("viridis",11)
img = plt.imshow(T, extent=[0, 1, 0, 1], aspect='auto', origin='lower',cmap=cmap)
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar(img, label="Temperatura")
plt.title("Heat map - neuman")
plt.xticks(ticks, ticks_labels)
plt.savefig(PATH + "heat_map-neuman.pdf")
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
cbar = plt.colorbar(sm, boundaries=np.linspace(0, 1, 10), label="čas")
# cbar.ax.invert_yaxis()
plt.xlim(0,1)
# plt.ylim(T.min(),T.max())
plt.title("Heat map - neuman")

plt.xticks(ticks, ticks_labels)
# plt.
plt.savefig(PATH + "heat_map-neuman2.pdf")


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