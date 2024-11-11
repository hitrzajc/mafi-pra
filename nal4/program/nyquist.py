import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time
from numpy.fft import *

plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
PATH = "../latex/pdfs/"
resize = lambda : plt.figure(figsize=(9, 3)) 

def f(x):
    return np.sin(x)

x = np.linspace(0,100,1000)
y = f(x)
resize()
plt.plot(x,y,color="black", label="$\sin(x)$")

xs = []
ys = []
dxs = []


fys, fxs = [], []

for i in range(6):
    n=1000
    x = np.linspace(0,n,n//(6)+(i-2)*n//30 + i//3)
    # n=100
    # x = np.linspace(0,n,n//(6)+(i-2)*3 + i//3)

    y = f(x)
    # xs.append(x)
    # ys.append(y)
    dx = x[1]-x[0]
    # print(x)
    # print(y)
    dxs.append(dx)
    plt.scatter(x,y,label="$f_c = {:.2f}$".format(1/dx),zorder=2)
    fy = fftshift(fft(y))/len(y)
    fx = fftshift(fftfreq(len(y), dx))
    fys.append(abs(fy))
    fxs.append(fx)
plt.xlim(0,100)
plt.legend()
plt.xlabel("$x$")
plt.title("Točke vzorčenja")
plt.savefig(PATH + "nyquist.pdf")
plt.clf()
resize()

for i in range(len(fxs)):
    plt.plot(fxs[i],fys[i],label="$f_c ={:.2f}$".format(1/dxs[i]))

x = np.linspace(0,1000,1000)
y = f(x)
x = fftshift(fftfreq(len(x), x[1]-x[0]))
y = fftshift(fft(y))/len(y)
plt.plot(x,abs(y),color="black",label="$FT(\sin(x))$")
# plt.xlim(fxs[-1][0]-0.1,fxs[-1][-1]+0.1)
plt.xlim(-0.17,0.17)
plt.legend()
plt.xlabel("$\\nu$")
plt.title("Točke preslikave")
plt.savefig(PATH + "nyquistfft.pdf")

plt.clf()