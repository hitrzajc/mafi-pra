import Methods
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
methods = [getattr(Methods,func) for func in dir(Methods) if callable(getattr(Methods, func)) and not func.startswith('__')]
PATH = "../latex/pdfs"

import numpy as np
PATH = "../latex/pdfs/"
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0

def dydt(t, y, k=0.1, T_zun =0):
    return -k * (y - T_zun)
step = 5
size = 5000
N = list(range(1,size,step))
y = {}

t = []
T_zacetna=100
for i in methods:
    y[i.__name__] = []


for n in tqdm(N):
    for i in range(step):
        # idx = n + i
        t.append(n+i)
    t = np.array(t)

    for method in methods:
        if method.__name__ == "rkf":
            start = time.time()
            method(dydt, T_zacetna)
            end = time.time()
        else:
            start = time.time()
            method(dydt, T_zacetna, t)
            end = time.time()
        y[method.__name__].append(end - start)

    # start = time.time()
    # method(dydt, T_zacetna)
    # end = time.time()

    t = list(t)

for i in methods:

    plt.plot(N, y[i.__name__],label = i.__name__)
plt.xlabel("Number of imput data")
plt.ylabel("t [s]")
plt.title("Časovna odvisnost hitrosti metod")
plt.legend()
plt.grid()
plt.yscale("log")
plt.savefig(PATH+"cas.pdf")