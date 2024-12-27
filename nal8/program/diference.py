#neskoncna jama

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy import linalg
a = 5
PATH = "../latex/pdfs/"
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0

def draw_potential(a):
    x = np.linspace(-a/2, a/2, 1000)
    y = np.zeros(1000)
    plt.plot(x, y, color="black")
    y_min, y_max = plt.gca().get_ylim()  
    plt.plot([-a/2, -a/2], [0, y_max], color='black', label="$V(x)$")
    plt.plot([a/2, a/2], [0, y_max], color='black')
    plt.ylim(y_min, y_max)
    
## diferencna metoda
N = 1000
h = a / (N + 1)  # Grid spacing
x = np.linspace(-a/2+h,a/2-h,N)
# h = x[1]-x[0]
diagonals = [-1 * np.ones(N-1), 2 * np.ones(N), -1 * np.ones(N-1)]
A = diags(diagonals, [-1, 0, 1]).toarray()
print(A[N-3:N, N-3:N])
print(A[:3][:3])

eigval, eigvec = linalg.eigh(A)
for i in range(N):
    norm = np.sqrt(np.sum(eigvec[:,i]**2) * h)
    eigval[i] = eigval[i] / h**2
    eigvec[:,i] = eigvec[:, i] / norm

analytical_eigval  = [(i * np.pi / a)**2 for i in range(1, N+1)]
analytical_eigval = np.array(analytical_eigval)
# print(eigval)

# print("Numerical Eigenvalues:", eigval[0:5])
# print("Analytical Eigenvalues:", analytical_eigval[:5])
for i in range(5):
    print(eigval[i]/analytical_eigval[i])
# print(norm, h, h**2, 1/h, 1/h**2)

################# TLE SE ZACNE RISATI
labels = ["$E_{} = {:.2e}".format(i+1, eigval[i]).replace('e', r'\times 10^{').replace('+0', '').replace('-', r'-') + '}$' for i in range(3)]
for i in range(3):
    plt.plot(x, eigvec[:, i], label=labels[i])

plt.legend()
ticks = np.linspace(-a/2, a/2, 5)
tick_labels = [r"$-\frac{a}{2}$", r"$-\frac{a}{4}$", r"$0$", r"$\frac{a}{4}$", r"$\frac{a}{2}$"]
plt.xticks(ticks, tick_labels)  # Set custom ticks and labels
plt.grid()
plt.title("Reišitve Schrödingerjeve enačbe za neskončno jamo z diferenčno metodo")
plt.xlabel("x")
plt.ylabel(r"$\psi(x)$")
draw_potential(a)
plt.savefig(PATH+"neskoncna_jama_num.pdf")

plt.clf()


########################################## une napake
# plt.xlabel("x")
# plt.ylabel(r"$\psi(x)$")
# plt.plot(x, eigvec[:, 0], label="prva rešitev")
# plt.plot(x, abs(eigvec[:, 1]), label="druga rešitev")
# plt.legend()
# plt.grid()
# plt.xticks(ticks, tick_labels)  # Set custom ticks and labels
# plt.title("Prve dve rešitvi, ki so napačne")
# # plt.yscale("log")
# plt.show()
# # plt.show()
# # draw_potential(a) 
# plt.yscale("log")
# plt.savefig(PATH+"neskoncna_jama_dif_napacne.pdf")
# plt.ylim(1e-20, 20)
# plt.savefig(PATH+"neskoncna_jama_dif_napacne_lim.pdf")
# plt.clf()
###################################

##############
## analiticna resitev
################
for i in range(1, 4):
    psi = np.sqrt(2/a) 
    kn = i * np.pi / a
    psi *= np.sin(kn*(x+a/2))
    # if i % 2 == 0:
    #     psi *= np.cos(kn*x)
    # else:
    #     psi *= np.sin(kn*x)
    plt.plot(x, psi, label=r"$E_{} = {:.2e}".format(i, analytical_eigval[i-1]).replace('e', r'\times 10^{').replace('+0', '').replace('-', r'-') + '}$')
plt.legend()
plt.grid()  
plt.xticks(ticks, tick_labels)  # Set custom ticks and labels
plt.xlabel("x")
plt.ylabel(r"$\psi(x)$")
plt.title("Analitične reišitve Schrödingerjeve enačbe za neskončno jamo")
draw_potential(a)
plt.savefig(PATH+"neskoncna_jama_analiticno.pdf")

plt.clf()




## absoltune napake

# N = 998
# kn = 1 * np.pi / a
# psi = np.sin(kn*(x+a/2)) * np.sqrt(2/a)
# plt.plot(x, psi,label="analitična")
# # plt.plot(x,eigvec[:, 2] / np.linalg.norm(eigvec[:,2]*h) ,label="numerična")
# # plt.plot(x,eigvec[:, 2] / np.sqrt(np.sum(eigvec[:,2]**2) * h),label="numerična")
# plt.legend()
# plt.show()
# plt.clf()
# exit()
mx = 0
mi = 100
Y =[ ]
for i in range(1, 4):
    psi = np.sqrt(2/a) 
    kn = i * np.pi / a
    psi *= np.sin(kn*(x+a/2))
    y = abs(psi**2 - eigvec[:, i-1]**2)
    mx = max(mx, max(y))
    mi = min(mi, min(y))
    plt.plot(x, y, label=r"$\Delta|\psi_{}|^2$".format(i))
draw_potential(a)
plt.xticks(ticks, tick_labels)  # Set custom ticks and labels
plt.grid()
plt.legend()
plt.title("Absolutne napake analitičnih rešitev")
plt.savefig(PATH+"neskoncna_jama_absolutne.pdf")
plt.clf()

mx = 0
mi = 100
Y =[ ]
from tqdm import tqdm

for i in tqdm(range(1, N+1)):
    psi = np.sqrt(2/a) 
    kn = i * np.pi / a
    psi *= np.sin(kn*(x+a/2))
    y = abs(psi**2 - eigvec[:, i-1]**2)
    mx = max(mx, max(y))
    mi = min(mi, min(y))
    Y.append(np.log(y))

from matplotlib.colors import LogNorm
plt.imshow(Y, aspect='auto', origin='lower', 
           extent=[-a/2, a/2, 0, len(Y)], cmap='viridis',vmin=-20, vmax=-50)
plt.colorbar(label="$\log$(absolutna napaka)")

plt.xticks(ticks, tick_labels)  # Set custom ticks and labels
# plt.show()
plt.ylabel("Stanje $\psi_{i}$, $i \in [1, N]$")
plt.title("Absolutne napake analitičnih rešitev")
plt.savefig(PATH+"neskoncna_jama_absolutne_barva.pdf")
plt.clf()

