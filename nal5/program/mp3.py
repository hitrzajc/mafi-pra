from pydub import AudioSegment
import numpy as np
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
from numpy.fft import *

PATH = "../latex/pdfs/"
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0

resize = lambda : plt.figure(figsize=(9, 2)) 


clovek = ['M', 'Z'] #moski zenska
glas = ['Glas 001','Glas 002']

pref = "./data/Sounds/"


# def visualize_fft(audio, title):
#     audio = AudioSegment.from_file(pref+sample_name+".m4a", format="m4a")
#     sample_rate = audio.frame_rate
#     plt.clf()
#     plt.xlabel()

color = ['#000000','#9f9f9f' ]
i=0
# resize()
for gender in clovek:
    
    # Load the m4a file
    audio = AudioSegment.from_file(pref+gender+".m4a", format="m4a")
    sample_rate = audio.frame_rate
    dt = 1/sample_rate
    # Convert audio to raw samples
    a = np.array(audio.get_array_of_samples())
    a = fftshift(fft(a))/len(a)
    # print(sum(a))
    a = abs(a) / sum(abs(a))
    x = fftshift(fftfreq(len(a),dt))
    plt.plot(x,a, color = color[i], label="spol: {}".format(gender))
    i+=1
plt.legend()
plt.xlabel("freqency [Hz]")
plt.xlim(50,20000)
plt.xscale("log")
plt.title("Gender freqencies")
plt.savefig(PATH+"fft_spol.pdf")
plt.clf()


for sample_name in glas:
    # Load the m4a file
    audio = AudioSegment.from_file(pref+sample_name+".m4a", format="m4a")
    sample_rate = audio.frame_rate
    # Convert audio to raw samples
    dt = 1/ sample_rate
    a = np.array(audio.get_array_of_samples())
    a = a / sum(a)

    
    a_std = np.std(a)
    for gender in clovek:
        audio = AudioSegment.from_file(pref+gender+".m4a", format="m4a")
        b = np.array(audio.get_array_of_samples())
        b = b / sum(b)
        b_std = np.std(b)
        y = correlate(a,b,method='fft',mode='full') #/ (a_std*b_std*len(b))
        
        # y = y/max(y)
        x = correlation_lags(len(a),len(b)) * dt
        resize()
        plt.plot(x,y, color = "black")
        
        my = max(y)
        idx = 0
        for i in range(len(y)):
            if y[i] == my:
                idx = x[i]
                break
        

        plt.scatter(idx,my,color='red',label="$({:.3f}\, ,{:.3f})$".format(idx,my))
        plt.legend()
        plt.title("Corelation between gender {} and sample {}".format(gender, sample_name))
        # plt.show()
        plt.xlabel("lag [s]")
        plt.savefig(PATH + "cor_{}_{}.pdf".format(gender, sample_name))
        plt.clf()
        # exit()

        

