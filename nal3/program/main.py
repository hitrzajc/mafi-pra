import numpy as np
from numpy import linalg
from Matrix import Matrix
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
λ = 0.01
N = 10
PATH = "../latex/pdfs/"

# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

###############################
#######   #spektri  ###########
###############################

# λs = np.linspace(0,1, 1000)
# N = 4
# for k in range(3):
#     ys = [[] for i in range(N)]
#     for λ in λs:
#         matrix = Matrix(N, λ)
#         matrix.fillings[k]()
#         sort = sorted(matrix.eigvals())
#         for i in range(N):
#             ys[i].append(sort[i])

#     for i in range(N):
#         plt.plot(λs, ys[i])
    
#     plt.xlabel("$\lambda$")
#     plt.ylabel("E")
#     plt.title("Spekter hamiltoniana tipa {}".format(k+1))
#     plt.grid()
#     plt.savefig(PATH+"spekter_{}.pdf".format(k+1), bbox_inches='tight', pad_inches=0)
#     plt.clf()

#########################################
##  Odvisnost priblizka od velikosti N ##
#########################################
# types = ["$\langle i|q|j \\rangle ^4$", "$\langle i|q^2|j\\rangle ^2$", "$\langle i|q^4|j \\rangle $"]
# λ = 0.1
# Ns = np.arange(4,100,1)
# for k in range(3):
#     ys = [[] for i in range(4)]
#     for i in tqdm(Ns):
#         matrix = Matrix(i, λ)
#         matrix.fillings[k]()
#         # eigs = matrix.eigh()[0][:4]
#         eigs = sorted(matrix.eigvals())[:4]
#         for j in range(len(eigs)):
#             ys[j].append(eigs[j])

#     for i in range(4):
#         plt.plot(Ns, np.array(ys[i])-ys[i][-1], label = "$E_{}$".format(i))
#     plt.legend()
#     plt.title(types[k])
#     plt.grid()
#     plt.xlabel("N")
#     plt.ylabel("$E-E_{\infty}$")
#     plt.savefig(PATH + "E(N){}.pdf".format(k),bbox_inches='tight', pad_inches=0)

#     plt.yscale("log")
#     plt.savefig(PATH + "log(E(N)){}.pdf".format(k),bbox_inches='tight', pad_inches=0)
#     plt.clf()

##############################
### CASOVNA ZAHTEVNOST #######
##############################
import time
algos = Matrix(1,1).algorithms
num_algos = len(algos)
ys = [[] for i in range(num_algos)]
N = 350
names = ["numpy.linalg.eigvals", "numpy.linalg.eig", "scipy.linalg.eigvals",
                           "QR_trid", "QR"]
Ns = np.arange(1,N+1,1)
for i in range(num_algos):
    for j in tqdm(range(1,N+1), desc="{}".format(algos[i].__name__)):
        tmp = Matrix(j, 0.1)
        tmp.fill1()
        start_time = time.time()
        tmp.use_alg(i)
        end_time = time.time()
        ys[i].append(end_time-start_time)
        if j == 110 and i == num_algos-2: #110
            break
        if j==50 and i == num_algos-1: #50
            break
    plt.plot(np.arange(1,len(ys[i])+1,1), ys[i], label = names[i])
plt.legend()
plt.ylabel("$t[ms]$")    
plt.xlabel("N")
plt.grid()

plt.savefig(PATH + "t(N).pdf",bbox_inches='tight', pad_inches=0)

plt.yscale("log")
plt.xscale("log")
# plt.show()
plt.savefig(PATH + "t(N)log.pdf",bbox_inches='tight', pad_inches=0)