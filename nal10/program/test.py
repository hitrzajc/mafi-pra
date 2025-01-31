import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm
from tqdm import tqdm
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


PATH = "../latex/pdf/"
# print(plt.rcParams.keys())
# plt.rcParams['savefig.bbox'] = 'tight'
# plt.rcParams['savefig.pad_inches'] = 0
plt.rcParams['savefig.dpi'] = 200
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

#TESTING
NNN = 0 #spremen v 0 za ne testiranje


times_t = [10, 50, 100, 300]
time_t = [[] for i in range(len(times_t))]

reds_colors = ["#ffe5e5","#ff9999","#ff4d4d","#cc0000"]
blues_colors = ["#cce5ff","#99ccff","#4da6ff","#0052cc"]


#blues_colors = ["#e5f2ff","#99ccff","#4da6ff","#0052cc"]

reds_cmap = LinearSegmentedColormap.from_list("MyReds", reds_colors, N=4)
blues_cmap = LinearSegmentedColormap.from_list("MyBlues", blues_colors, N=4)

# # Register custom colormaps
# cm.register_cmap(cmap=reds_cmap)
# cm.register_cmap(cmap=blues_cmap)

# Sample color arrays from the new colormaps
reds = reds_cmap(np.linspace(0, 1, 4))
blues = blues_cmap(np.linspace(0, 1, 4))
fig = plt.figure()
# ax1 = fig.add_subplot(111)
ax1 = fig.gca()

fig.set_size_inches(8, 6)
# ax = fig.add_subplot(111)

ax1.yaxis.label.set_color('deepskyblue')
ax1.tick_params(axis='y', colors='deepskyblue')
ax1.set_ylabel('Time [s]', color='deepskyblue')

for k in tqdm(range(len(times_t)-NNN)):
    t_siz = times_t[k]
    xxx = []
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
        time_t[k].append(end - start)
        xxx.append(x_siz)

    ax1.plot(xxx, time_t[k],color=blues[k],label="$N_t = $"+str(t_siz))

# Make left side of plot y-axis lightblue color with numbers blue as well
ax1.grid(axis='x')



times_x = [10, 50, 100, 300]
time_x = [[] for i in range(len(times_x))]
# plt.legend()
# blues = colormaps.get_cmap('Blues')

# make right-side y-axis red with red numbers
ax2 = ax1.twinx()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Time [s]', color='salmon')
ax2.yaxis.label.set_color('salmon')
ax2.tick_params(axis='y', colors='salmon')



for k in tqdm(range(len(times_x)-NNN)):
    x_siz = times_x[k]
    xxx = []
    for t_siz in range(5,Nt,5):
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
        time_x[k].append(end - start)
        xxx.append(t_siz)
    ax2.plot(xxx, time_x[k], color=reds[k], label="$N_x = $"+str(x_siz))

# ax = fig.add_subplot(111)

# 
norm1 = plt.Normalize(vmin=min(times_t), vmax=max(times_t))
norm2 = plt.Normalize(vmin=min(times_x), vmax=max(times_x))

sm1 = cm.ScalarMappable(cmap=blues_cmap,norm=norm1)
sm2 = cm.ScalarMappable(cmap=reds_cmap ,norm=norm2)

# Create separate axes for colorbars so the main plot retains its size
pos1 = ax1.get_position()
ax1.set_position([pos1.x0 + 0.1, pos1.y0, pos1.width - 0.2, pos1.height])  # Shift right for cbar1
ax2.set_position([pos1.x0 + 0.1, pos1.y0, pos1.width - 0.2, pos1.height])  # Sync ax2

cbar1_ax = fig.add_axes([0.1, pos1.y0, 0.02, pos1.height])  # Left colorbar
cbar2_ax = fig.add_axes([0.90, pos1.y0, 0.02, pos1.height])  # Right colorbar (shifted more to the right)


cbar1 = plt.colorbar(sm1, cax=cbar1_ax)
cbar1.set_label('$N_t$')
cbar1.ax.yaxis.set_ticks_position('left')
cbar1.ax.yaxis.set_label_position('left')
LOL = np.linspace(10,300,4,endpoint=False)

cbar1.set_ticks(LOL+(LOL[1]-LOL[0])/2)  # Set the ticks at specific values
cbar1.set_ticklabels([str(t) for t in times_t])  # Set the tick labels

for i in range(len(cbar1.ax.get_yticklabels())):
    color = blues_cmap(i/4)  # Get the color from the colormap for the tick value
    label = cbar1.ax.yaxis.get_ticklabels()[i]  # Get the tick label
    label.set_color(color)  # Set the color of the label to match the color of the tick value

cbar2 = plt.colorbar(sm2, cax=cbar2_ax)
cbar2.set_label('$N_x$')
cbar2.set_ticks(LOL+(LOL[1]-LOL[0])/2)  # Set the ticks at specific values
cbar2.set_ticklabels([str(t) for t in times_t])  # Set the tick labels

for i in range(len(cbar2.ax.get_yticklabels())):
    color = reds_cmap(i/4)  # Get the color from the colormap for the tick value
    label = cbar2.ax.yaxis.get_ticklabels()[i]  # Get the tick label
    label.set_color(color)  # Set the color of the label to match the color of the tick value


# ax2.set_axisbelow(True)  # Ensure grid lines are behind the plotted data
# ax2.xaxis.grid(True, which='major', color='grey')  # Only vertical lines
ax1.set_xlabel("Število točk v dimenziji x (modra), t(rdeča)")
# plt.ylabel("Time [s]")
ax1.set_title("Časovna zahtevnost")
ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
# print(y)
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin,ymax)
# plt.show()

plt.savefig(PATH + "time1.pdf")


