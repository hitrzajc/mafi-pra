from Path import Path, Paths
import numpy as np
from tqdm import tqdm
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

PATH = "../latex/pdfs/"


###3### prvi del

N = 100000
μs = [1.01, 2, 3, 10, 15, 30]
for i in μs:
    test = Path(100000,i)
    test.generate()
    test.draw(-0.30, 2.75, -0.5, 4.5)
###############


### Dugi del
N = 1000
path_len = 1000
# for i in tqdm(μs):
μ = 2
maksL = np.inf
minL = 0.01
paths = Paths(N, path_len, μ, maksL, minL)
paths.generate()
paths.flight2()
##############

#### Tretji Del
μs = np.linspace(1.11, 4, 50)
y = []
y1 = []
def γ(x):
    if x==2:
        return 2
    if 1<x and x<3:
        return 2/(x-1)
    if x>3:
        return 1

N = 100
path_len = 1000
maksL = 1000000
minL = 0.01
for μ in tqdm(μs):
    paths = Paths(N,path_len,μ,maksL,minL)
    paths.generate()
    y.append(abs(paths.flight() - γ(μ)))

plt.xlabel("$\mu$")
plt.ylabel("$\Delta \gamma (\mu)$")
plt.plot(μs, y)
plt.title("Absolutna razlika med teoretično in simulirano vrednostjo")
plt.savefig(PATH+"flights.pdf",bbox_inches='tight', pad_inches=0)

############cetrti del
N = 500
path_len = 2000
# for i in tqdm(μs):
μ = 2
maksL = np.inf
minL = 0.01
paths = Paths(N, path_len, μ, maksL, minL)
paths.generate()
paths.walk2()

#peti del
μs = np.linspace(1.8, 5, 50)
y = []
y1 = []
def γ(x):
    if 1<x and x<2:
        return 2
    if 2<=x and x<3:
        return 4-x
    if x>=3:
        return 1

N = 100
path_len = 1000
maksL = np.inf
minL = 0.01
y1 = []
y2 = []
for μ in tqdm(μs):
    paths = Paths(N,path_len+500,μ,maksL,minL)
    paths.generate()
    tmpP = paths.walk()
    tmpG = γ(μ)
    y1.append(tmpP)
    y2.append(tmpG)
    y.append(abs(tmpP - γ(μ)))



plt.xlabel("$\mu$")
plt.ylabel("$\Delta \gamma (\mu)$")
plt.plot(μs, y)
plt.title("Absolutna razlika med teoretično in simulirano vrednostjo")
plt.savefig(PATH+"walks.pdf",bbox_inches='tight', pad_inches=0)
plt.clf()

plt.plot(μs, y1, label = "Simuliran $\\nu$($\mu$)")
plt.plot(μs, y2, linestyle = "dashed", label = "Teoretična vrednost")
plt.xlabel("$\mu$")
plt.ylabel("$\gamma (\mu)$")
plt.savefig(PATH+"walkss.pdf",bbox_inches='tight', pad_inches=0)
