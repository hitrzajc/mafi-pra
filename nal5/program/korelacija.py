import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

PATH = "../latex/pdfs/"
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0

resize = lambda : plt.figure(figsize=(9, 2)) 



sova_file = ['bubomono.txt','bubo2mono.txt']
sova_data = {}
sample_rate = 44100
dt = 1/sample_rate
for file_name in sova_file:
    with open("./data/"+file_name,'r') as file:
        sova_data[file_name] = np.array([int(i) for i in file.readlines()])


mix_file = ['mix.txt','mix1.txt','mix2.txt','mix22.txt']
mix_data = {}

for file_name in mix_file:
    with open("./data/"+file_name,'r') as file:
        mix_data[file_name] = np.array([int(i) for i in file.readlines()])

########### EN DEL
for mix in mix_file:
    a = mix_data[mix]
    a_std = np.std(a)
    for sova in sova_file:
        b = sova_data[sova]
        b_std = np.std(b)
        y = correlate(a,b,method='fft')/(a_std*b_std*len(b))
        # y = y/max(y)
        x = np.arange(-len(b)+1,len(a))*dt
        resize()

        my = max(y)
        idx = 0
        for i in range(len(y)):
            if y[i] == my:
                idx = x[i]
                break
        

        plt.scatter(idx,my,color='red',label="$({:.3f}\, ,{:.3f})$".format(idx,my))

        plt.title("Corelation for {} and {}".format(mix,sova))
        plt.plot(x,y,color="black")
        plt.xlabel("lag [s]")
        plt.legend()
        plt.savefig(PATH+"cor_{}_{}.pdf".format(mix,sova))
        # plt.show()
        plt.clf()


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


for sova in sova_file:
    a = np.array(sova_data[sova])
    a = a / max(a)
    N = len(a)
    a_std = np.std(a)
    y = correlate(a,a,method='fft')/(a_std*a_std*len(a))
    x = np.arange(-N+1,N)*dt
    # for i in range(len(x)):
    #     y[i] = y[i]/((N-x[i])**2)
    plt.xlabel("Sample offset [s]")
    plt.plot(x,y,color = "black")
    plt.title("Autocorelation "+sova)
    # plt.show()
    plt.savefig(PATH+sova+"_acor.pdf")
    plt.clf()