import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time


plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
PATH = "../latex/pdfs/"

def fft(y): #dx je (sampling period)/(number of data points)
    N = len(y)
    H = []
    for n in range(N):
        Hn = 0
        for k in range(N):
            Hn += y[k]*np.exp(2j*np.pi*k*n/N)
        H.append(Hn)
    H = np.array(H)
    H = np.fft.fftshift(H)/N
    # H = np.roll(H,int(N/2))/N

    return H

def g(x):
    return np.cos(np.sin(np.sqrt(abs(x))))*np.sin(x) - np.sin(0.3*x)
    # return np.sin(3*x)
    # return np.exp(-x*x)


##############################
#### NEKA FUNKCIJA K NI GAUSS
##############################

resize = lambda : plt.figure(figsize=(9, 3)) 


x1 = np.linspace(0,100,10000)
dx = x1[1] - x1[0]
y1 = g(x1)
# y2 = np.abs(fft(y1))
y2 = np.fft.fftshift(np.fft.fft(y1))/len(y1)
x2 = np.fft.fftshift(np.fft.fftfreq(len(x1),dx))

####PRED
plt.plot(x1,y1,label="$f(x) = \cos (\sin (\sqrt{x}))\sin(x) - \sin (0.3 x)$")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.legend()
plt.title("Pred transformacijo")
plt.savefig(PATH+"pred.pdf")
plt.clf()


x1 = np.linspace(0,10000//2,100000)
dx = x1[1] - x1[0]
y1 = g(x1)
# y2 = np.abs(fft(y1))
y2 = np.fft.fftshift(np.fft.fft(y1))/len(y1)
x2 = np.fft.fftshift(np.fft.fftfreq(len(x1),dx))
####PO
resize()
plt.plot(x2,np.abs(y2),label="$|FT(f)(\\nu)|$")
plt.xlabel("$\\nu$")
# plt.ylabel("$|FT(f)|$")
plt.legend()
plt.xlim(-0.25,0.25)
plt.title("Po transformaciji")
plt.savefig(PATH+"po.pdf")
plt.clf()

###IMAG
resize()
plt.plot(x2,np.imag(y2),label="$\mathfrak{Im}FT(f)(\\nu)$")
plt.xlabel("$\\nu$")
# plt.ylabel("$\mathfrak{Im}FT(f)$")
plt.legend()
plt.xlim(-0.25,0.25)
plt.title("Po transformaciji")
plt.savefig(PATH+"imag.pdf")
plt.clf()
###REAL
resize()
plt.plot(x2,np.real(y2),label="$\mathfrak{Re}FT(f)(\\nu)$")
plt.xlabel("$\\nu$")
# plt.ylabel("$\mathfrak{Re}FT(f)$")
plt.legend()
plt.xlim(-0.25,0.25)
plt.title("Po transformaciji")
plt.savefig(PATH+"real.pdf")
plt.clf()
# exit()
######################
### CASOVNA ODVISNOST
######################

N = 1000
y = [[],[]]
x = [0.1,0.2,]
for n in tqdm(range(3,N)):
    x.append(random.random())
    start_t = time.time()
    fft(x)
    end_t = time.time()
    y[0].append(end_t-start_t)

    start_t = time.time()
    np.fft.fft(x)
    end_t = time.time()
    y[1].append(end_t-start_t)
x = list(range(3,N))
plt.plot(x,y[0],label="Naivna $\mathcal{O}(n^2)$ implementacija")
plt.plot(x,y[1],label="Numpy $\mathcal{O}(n\log n)$ implementacija")
plt.yscale("log")
plt.ylabel("t[s]")
plt.xlabel("N")
plt.legend()
plt.grid()
plt.title("Časovna odvisnost implementacije")
plt.savefig(PATH+"t(N).pdf")
# plt.show()