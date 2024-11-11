import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
PATH = "../latex/pdfs/"
def f(x,mu=0,sigma=1): #gauss function
    return np.exp(-(x-mu)**2/(2*sigma ** 2))

x = np.linspace(-10,10,300)
y = f(x)
dx = x[1] - x[0]  # Sampling interval
freq = np.fft.fftfreq(len(x), d=dx)

plt.plot(x,y)
plt.title("Initial gauss")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.savefig(PATH+"gauss.pdf")
plt.clf()

fy = np.fft.fft(y)
sfy = np.fft.fftshift(abs(fy))/np.sqrt(len(x))
iy = abs(np.fft.ifft(fy))

plt.plot(np.fft.fftshift(freq),sfy)
plt.title("FT gauss")
plt.xlabel("$\\nu$")
plt.ylabel("$FFT(f)(\\nu)$")
plt.savefig(PATH+"fft(gauss).pdf")
plt.clf()

plt.title("Inverse Ft gauss")
plt.plot(x,iy)
plt.xlabel("$x$")
plt.ylabel("$FFT^{-1}(FFT(f))(x)")
plt.savefig(PATH+"inf(fft).pdf")
plt.clf()

plt.title("absolutna napaka pred tranformacijo in po inverzu")
plt.xlabel("$x$")
plt.ylabel("$\Delta f$")
plt.plot(x,abs(iy-y))
plt.savefig(PATH+"abs_napaka.pdf")