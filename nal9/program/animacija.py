import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
#pde for diffusion equation 1D
N = 1000
x = np.linspace(0, 1, N)
dx = x[1]-x[0]
t = np.linspace(0, 10, N)
# T0 = np.sin(np.pi * 10 *x)
T0 = np.exp(-100*(x-0.5)**2)


# Tk0 = fft.fft(T0,N)
# fk = fft.fftfreq(n=N,d=dx)

Tk = np.zeros((N,N),dtype=complex)
Tk0 = fft.fft(T0,N)#/np.sqrt(N)

fk = fft.fftfreq(n=N,d=dx)
# fk = fft.fftshift(fk)
Tk[0] = Tk0
h = t[1]-t[0]
D = 1
T = np.zeros((N,N))
T[0] = T0
a = x[-1] - x[0]
plt.plot(fk,fft.fftshift(abs(Tk[0])))
plt.show()
# for i in range(1,len(t)):
#     for j in range(len(x)):
#         Tk[i][j] = Tk[i-1][j] + h*D*(-4*np.pi**2*(j/N)**2) * Tk[i-1][j]
#         if abs(1-h*D*(4*np.pi**2*(j/N)**2)) >= 1:
#             print(abs(1-h*D*(-4*np.pi**2*(j/N)**2)))
    # T[i] = fft.ifft(Tk[i])


for j in range(len(x)):
    Tk[:,j] = np.exp(-4*np.pi**2*D*(fk**2)*t)*Tk0[j]
    T[:,j] = fft.ifft(Tk[:,j])
    # plt.plot(fk,abs(Tk[i]))
    # plt.show()
    # break

plt.plot(x,T[0])
plt.plot(x,T[1])
plt.plot(x,T[500])
plt.show()

# exit()
plt.imshow(T, aspect='auto', cmap=cm.coolwarm)
plt.colorbar()
plt.show()



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
# Create animation
ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)

# Save as a GIF
ani.save("animation.gif", writer=PillowWriter(fps=10))