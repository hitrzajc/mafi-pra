import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time


plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
PATH = "../latex/pdfs/"

def correlation_FFT(f, g):
    if len(f) > len(g):
        g = np.concatenate((g, np.zeros(len(f) - len(g))))
    elif len(f) < len(g):
        f = np.concatenate((f, np.zeros(len(g) - len(f))))

    f = np.asarray(f)
    g = np.asarray(g)
    N = len(f)

    f = np.concatenate((f, np.zeros(N)))
    g = np.concatenate((g, np.zeros(N)))

    ns = np.arange(-N, N)

    F = np.fft.fft(f)/N
    G = np.fft.fft(g)/N

    temp = np.conjugate(F) * G

    temp = np.fft.ifft(temp).real

    correlation = np.fft.ifftshift(temp)

    return ns, correlation


def non_FT_Corelation(f:list,g:list):
    N = len(g)
    out1,out2 = [],[]
    for n in range(N):
        sum = 0
        for i in range(N):
            try:
                sum += g[i+n] * f[i]
            except IndexError:
                break

        out1.append(sum / (N-n))
        out2.append(sum/N)
    return out1,out2

from random import random
step = 5
a = [1]*step
x = []
y = [[],[]]
for i in tqdm(range(0, 2000, step)):
    x.append(i+step)
    for i in range(step):
        a.append(random())
    start = time.time()
    non_FT_Corelation(a,a)
    end = time.time()
    y[0].append(end-start)

    start = time.time()
    correlation_FFT(a,a)
    end = time.time()
    y[1].append(end-start)


plt.plot(x,y[0],label="brez fft")
plt.plot(x,y[1],label="s fft")
plt.xlabel("N")
plt.ylabel("t [s]")
plt.legend()
plt.grid()
plt.title("Časovna odvisnost algoritmov")
plt.savefig(PATH+"cas-lin.pdf")
plt.yscale("log")
plt.savefig(PATH+"cas-log.pdf")
