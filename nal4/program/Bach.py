import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *
from tqdm import tqdm

import os

# Folder path
PATH = '../latex/pdfs/'
folder_path = './data/txt/'
resize = lambda : plt.figure(figsize=(9, 2)) 

# Dictionary to store file content with file names
file_contents = {}

# Loop through each file in the directory
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as file:
        # Read file content
        content = file.readlines()
        # Store content with filename as the key
        file_contents[filename] = content

# Display file contents
# for name, lines in file_contents.items():
#     print(f"Filename: {name}")
#     print("Content:", ''.join(lines))


for name, lines in file_contents.items():
    y = []
    for line in tqdm(lines):
        y.append(int(line))
    # print(name[5:])
    freq = 1/int(name[5:])

    y = fftshift(fft(y))/len(y)
    x = fftshift(fftfreq(n=len(y), d=freq))
    resize()

    plt.plot(x,abs(y))
    plt.xlabel("$\\nu\,[Hz]$")
    plt.ylabel("$|F|$")
    plt.title(name+"Hz")
    plt.savefig(PATH+name[5:]+".pdf")
    plt.clf()
    
    ##real
    resize()
    plt.plot(x,y.real)
    plt.xlabel("$\\nu\,[Hz]$")
    plt.ylabel("$\mathfrak{Re}\,F$")
    plt.title(name+"Hz")
    plt.savefig(PATH+name[5:]+"Re.pdf")
    plt.clf()

    ##imag
    resize()
    plt.plot(x,y.imag)
    plt.xlabel("$\\nu\,[Hz]$")
    plt.ylabel("$\mathfrak{Im}\,F$")
    plt.title(name+"Hz")
    plt.savefig(PATH+name[5:]+"Im.pdf")
    plt.clf()


    # plt.show()
    # exit()