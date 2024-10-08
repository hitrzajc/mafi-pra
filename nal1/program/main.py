import matplotlib.pyplot as plt
from mpmath import *
import numpy as np
from scipy import special
from tqdm import tqdm

mp.dps = 100
# eps = mpf(1e-300)

mp.pretty = True

data_points = 1001
min_x = -30
max_x = 30
interval = (max_x-min_x)/data_points
data_x = []
mininterval=0

for i in range(data_points):
    # data_x.append(Decimal(min_x + i*interval))
    data_x.append(mpf(min_x + i*interval))


Ai = []
Bi = []
ysmallAi = []
ysmallBi = []
ybigAi = []
ybigBi = []
#tocne Ai in Bi
def get_exact():
    for i in tqdm(data_x,mininterval=mininterval):
        Ai.append(airyai(i))
        Bi.append(airybi(i))
  
def draw_airy():
    # get_exact()
    plt.plot(data_x, Ai, label = "Ai(x)")
    plt.plot(data_x, Bi, label = "Bi(x)")
    plt.xlim(-15,5)
    plt.ylim(-0.6,0.6)
    plt.axhline(y=0,color='b',linestyle='--',linewidth=1)
    plt.legend()
    plt.xlabel("x")
    plt.xticks(range(-15,6,5))
    # plt.show()
    plt.savefig("./pdfs/True_airy.pdf")
    plt.clf()


def get_small():
    for i in tqdm(data_x,mininterval=mininterval):
        ai, bi = small_airy(i)
        ysmallAi.append(ai)
        ysmallBi.append(bi)

def draw_small_airy():
    # get_small()
    plt.plot(data_x, ysmallAi, label = "Ai(x)")
    plt.plot(data_x, ysmallBi, label = "Bi(x)")
    plt.xlim(-15,5)

    plt.ylim(-0.6,0.6)
    plt.axhline(y=0,color='b',linestyle='--',linewidth=1)
    plt.legend()
    plt.xlabel("x")
    plt.xticks(range(-15,6,5))
    # plt.show()
    plt.savefig("./pdfs/small_airy.pdf")
    plt.clf()
    
#returns Ai(x), Bi(x)
def small_airy(x):
    α = mpf(0.35502805388781723)
    β = mpf(0.25881940379280679)
    sqrt3 = sqrt(3)
    # fac13 = mpf(0.89297951156924921)
    # fac23 = mpf(0.90274529295093361)

    def f(x):
        k = 1
        ans = mpf(1)
        a = mpf(1)
        while True:
            prev = a
            a *= 3 * x*x*x
            for i in range(3):
                a/=(3*k-i)
            
            a *= (1/3 + k - 1)
            k+=1
            if abs(prev -a)<eps and a<eps:
            # if prev==a and k==1000:
                return ans
            ans += a
    def g(x):
        k = 1
        a = mpf(x)
        ans = mpf(x)
        while True:
            prev = a
            a *= 3 * x*x*x
            a *= (2/3 + k -1)
            for i in range(3):
                a/= (3*k+1-i)
            k+=1
            if abs(prev -a)<eps and a<eps:
            # if prev==a and k>1000:
                return ans
            ans += a
            
    return α*f(x)-β*g(x), sqrt3 * (α*f(x)+β*g(x))

def L(x):
    ans = mpf(1)
    a = mpf(1)
    s = mpf(1)
    while True:
        prev = a
        for i in range(3):
            a *= mpf(3*s - 1/2 - i)
        a/=mpf(s-1/2)
        a/=mpf(s)
        
        a/=mpf(54)
        a/=mpf(x)
        
        # if prev < a: # or s == 16
        if s==17:
            return ans
        ans += a
        s+=1

def P(x):
    ans = mpf(1)
    a = mpf(1)
    s = mpf(1)
    x = mpf(x)
    while True:
        a = -a
        prev = a
        a /= mpf(54)*mpf(54)
        for i in range(6):
            a *= mpf(6*s-1/2-i)
        a /= (2*s-1/2)*(2*s-1/2-1)
        a /= (2*s)*(2*s-1)
        a /= mpf(x)*mpf(x)
        if abs(prev) < abs(a):
        # if s == 22:
            return ans
        ans += a
        s+=1

def Q(x):
    ans = mpf(0)
    a = mpf((3-1/2))*mpf((2-1/2))/54/x
    ans += a
    s = mpf(1)
    
    while True:
        prev = a
        a = -a
        for i in range(6):
            a *= (6*s+3-1/2-i)
        a/=mpf(2*s+1)*mpf(2*s)
        a/=mpf(2*s+1/2)*mpf(2*s-1/2)
        a/=mpf(54)*mpf(54)
        a/=mpf(x)*mpf(x)
        if abs(prev)<abs(a):
            return ans
        ans+=a
        s+=1

def big_airy(x): #tole je za popravt se
    tmpAi = tmpBi = 0
    ksi = mpf(2/3)*abs(x)**mpf(3/2)
    if x > 0: 
        tmpAi =  mpf(np.e)**(-ksi) / (mpf(2) * mpf(np.pi)**mpf(1/2) * x**mpf(1/4)) * L(-ksi)
        tmpBi = mpf(np.e)**(ksi) / (mpf(np.pi)**mpf(1/2) * x**mpf(1/4)) * L(ksi)
    
    if x < 0:
        tmpAi = 1/(mpf(np.pi)**mpf(1/2) * (-x)**mpf(1/4)) * (sin(ksi - mpf(np.pi/4)) * Q(ksi) + cos(ksi - mpf(np.pi/4)) * P(ksi))
        tmpBi =  1/(mpf(np.pi)**mpf(1/2) * (-x)**mpf(1/4)) * (-sin(ksi - mpf(np.pi/4)) * P(ksi) + cos(ksi - mpf(np.pi/4)) * Q(ksi))

    return tmpAi, tmpBi    

def get_big():
    for i in tqdm(data_x,mininterval=mininterval):
        ai, bi = big_airy(i)
        ybigAi.append(ai)
        ybigBi.append(bi)

def draw_big_airy():
    # get_big()
    plt.plot(data_x, ybigAi, label = "Ai(x)")
    plt.plot(data_x, ybigBi, label = "Bi(x)")
    plt.xlim(-15,5)
    # plt.xlim(-15,20)

    plt.ylim(-0.6,0.6)
    plt.axhline(y=0,color='b',linestyle='--',linewidth=1)
    plt.legend()
    plt.xlabel("x")
    plt.xticks(range(-15,6,5))
    # plt.show()
    plt.savefig("./pdfs/big_airy.pdf")
    plt.clf()

def draw_absolute():
    taylorAi = abs(np.array(Ai) - np.array(ysmallAi))
    asimAi = abs(np.array(Ai) - np.array(ybigAi))
    # smallBi = abs(np(Bi) - np(ysmallBi))

    plt.plot(data_x, taylorAi, label = "$\Delta Ai_{taylor}(x)$")
    plt.plot(data_x, asimAi, label = "$\Delta Ai_{asim}$")
    plt.yscale("log")
    plt.axhline(y=1,color='b',linestyle='--',linewidth=1,label="1")
    plt.axhline(y=1e-10,color='r',linestyle='--',linewidth=1,label="$10^{-10}$")
    plt.legend()
    plt.savefig("./pdfs/Ai_abs_error.pdf")
    plt.clf()

    taylorAi = abs(np.array(Bi) - np.array(ysmallBi))
    asimAi = abs(np.array(Bi) - np.array(ybigBi))
    plt.plot(data_x, taylorAi, label = "$\Delta Bi_{taylor}(x)$")
    plt.plot(data_x, asimAi, label = "$\Delta Bi_{asim}(x)$")
    plt.yscale("log")
    
    plt.axhline(y=1,color='b',linestyle='--',linewidth=1,label="1")
    plt.axhline(y=1e-10,color='r',linestyle='--',linewidth=1,label="$10^{-10}$")

    plt.legend()
    plt.savefig("./pdfs/Bi_abs_error.pdf")
    plt.clf()

def draw_relative():
    taylorAi = abs(1-np.array(ysmallAi)/np.array(Ai))
    asimAi = abs(1-np.array(ybigAi)/np.array(Ai))
    
    # smallBi = abs(np(Bi) - np(ysmallBi))

    plt.plot(data_x, taylorAi, label = "relaitve $Ai_{taylor}(x)$")
    plt.plot(data_x, asimAi, label = "relaitve $Ai_{asim}(x)$")
    plt.yscale("log")
    plt.axhline(y=1,color='b',linestyle='--',linewidth=1,label="1")
    plt.axhline(y=1e-10,color='r',linestyle='--',linewidth=1,label="$10^{-10}$")
    plt.legend()
    plt.savefig("./pdfs/Ai_rel_error.pdf")
    plt.clf()

    taylorAi = abs(1-np.array(ysmallBi)/np.array(Bi))
    asimAi = abs(1-np.array(ybigBi)/np.array(Bi))
    plt.plot(data_x, taylorAi, label = "relaitve $Bi_{taylor}(x)$")
    plt.plot(data_x, asimAi, label = "relaitve $Bi_{asim}(x)$")
    plt.yscale("log")
    
    plt.axhline(y=1,color='b',linestyle='--',linewidth=1,label="1")
    plt.axhline(y=1e-10,color='r',linestyle='--',linewidth=1,label="$10^{-10}$")

    plt.legend()
    plt.savefig("./pdfs/Bi_rel_error.pdf")
    plt.clf()
# exit()
import concurrent.futures
pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
pool.submit(get_big)
pool.submit(get_exact)
pool.submit(get_small)
pool.shutdown(wait=True)

# exit()
draw_airy()
draw_absolute()
draw_small_airy()
draw_big_airy()
draw_relative()
