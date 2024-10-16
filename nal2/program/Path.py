import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit

PATH = "../latex/pdfs/"
class Paths():
    def generate(self):
        # import concurrent.futures
        # pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        for i in range(self.Npaths):
            self.paths.append(Path(self.path_len, self.μ, self.b,self.a,seed = self.seed))
            # pool.submit(self.paths[i].generate())
            self.paths[i].generate()
            self.seed += 1
        
        # pool.shutdown(wait=True)
    # za isti t = step mogledas MAD
    # za mi = 2
    def flight2(self): #vse dolzine trajajo enako dolgo
        tmpR = [0 for i in range(self.Npaths)]
        sigma2 = []
        x = []
        for i in range(self.path_len-1):
            for p in range(self.Npaths):
                tmpR[p] = self.paths[p].r[i]
            x.append(i)
            MAD = median_abs_deviation(tmpR)
            sigma2.append(MAD**2/0.45494)

        plt.plot(x,sigma2)
        plt.xlabel("$t$")
        plt.ylabel("$\sigma^2(t)$")
        plt.title("flight")
        plt.savefig(PATH+"flight.pdf",bbox_inches='tight', pad_inches=0)

        plt.title("flight with log scales")
        plt.clf()
        x = np.log(np.array(x[1:]))
        sigma2 = np.log(np.array(sigma2[1:]))

        def linearf(x,m,b):
            return m*x + b
        popt, pcov = curve_fit(linearf, x, sigma2)

        plt.scatter(x,sigma2)
        plt.plot(x, linearf(x, *popt), label='Fitted function\n $f(\log(t)) = k t + c$\n$k={:.3f} \pm {:.3f}$'.format(popt[0],np.sqrt(np.diag(pcov))[0]), color='red')
        plt.xlabel("$\log(t)$")
        plt.ylabel("$\log(\sigma^2(t))$")
        plt.legend()
        plt.savefig(PATH+"flight_log.pdf",bbox_inches='tight', pad_inches=0)
        plt.clf()

    def flight(self):
        tmpR = [0 for i in range(self.Npaths)]
        sigma2 = []
        x = []
        for i in range(self.path_len-1):
            for p in range(self.Npaths):
                tmpR[p] = self.paths[p].r[i]
            x.append(i)
            MAD = median_abs_deviation(tmpR)
            sigma2.append(MAD**2/0.45494)
        
        x = np.log(np.array(x[1:]))
        sigma2 = np.log(np.array(sigma2[1:]))
        popt, pcov = curve_fit(lambda x,γ,c: γ * x + c, x, sigma2)
        # popt, pcov = curve_fit(lambda x,γ,c: pow(x,γ) + c, x, sigma2)
        return popt[0]

    def walk2(self):
        v = self.a # v = s/t
        tmpR = [0 for i in range(self.Npaths)]
        # total_distance = [0 for i in range(self.Npaths)]
        curent_idx = [0 for i in range(self.Npaths)]
        strli_cez = [0 for i in range(self.Npaths)]
        sigma2 = []
        x = np.linspace(0,1000,1000) #cas
        for j in range(1,len(x)): #biksa negativne strli_cez alii pac?
            for i in range(self.Npaths):
                ds = v#(x[j] - x[j-1])*v
                #v*t tle mormo bit
                k = curent_idx[i]
                d = abs(self.paths[i].r[k] - self.paths[i].r[k+1])
                if d-strli_cez[i] <= ds:
                    ds-=d-strli_cez[i]
                    k+=1
                    d = abs(self.paths[i].r[k] - self.paths[i].r[k+1])
                    while  d <= ds: #preveri k index
                        ds -= d
                        k+=1
                        d = abs(self.paths[i].r[k] - self.paths[i].r[k+1])
                    strli_cez[i] = ds
                
                else:
                    strli_cez[i] += ds

                curent_idx[i] = k
                tmpR[i] = strli_cez[i] + self.paths[i].r[k]
            
            MAD = median_abs_deviation(tmpR)
            sigma2.append(MAD**2/0.45494)
        x = x[1:]
        plt.plot(x,sigma2)
        plt.xlabel("$t$")
        plt.ylabel("$\sigma^2(t)$")
        plt.title("walk")
        plt.savefig(PATH+"walk.pdf",bbox_inches='tight', pad_inches=0)

        plt.title("walk with log scales")
        plt.clf()
        x = np.log(np.array(x[10:]))
        sigma2 = np.log(np.array(sigma2[10:]))

        def linearf(x,m,b):
            return m*x + b
        popt, pcov = curve_fit(linearf, x, sigma2)

        plt.scatter(x,sigma2)
        plt.plot(x, linearf(x, *popt), label='Fitted function\n $f(\log(t)) = k t + c$\n$k={:.3f} \pm {:.3f}$'.format(popt[0],np.sqrt(np.diag(pcov))[0]), color='red')
        plt.xlabel("$\log(t)$")
        plt.ylabel("$\log(\sigma^2(t))$")
        plt.legend()
        plt.savefig(PATH+"walk_log.pdf",bbox_inches='tight', pad_inches=0)
        plt.clf()
  
    def walk(self):
        v = self.a #v=s/t
        tmpR = [0 for i in range(self.Npaths)]
        # total_distance = [0 for i in range(self.Npaths)]
        curent_idx = [0 for i in range(self.Npaths)]
        strli_cez = [0 for i in range(self.Npaths)]
        sigma2 = []
        x = np.linspace(0,1000 ,1000)
        for j in range(1,len(x)): #biksa negativne za rikverc d strli_cez
            for i in range(self.Npaths):
                ds = v#(x[j] - x[j-1])*v
                #v*t tle mormo bit
                k = curent_idx[i]
                d = abs(self.paths[i].r[k] - self.paths[i].r[k+1])
                if d-strli_cez[i] <= ds:
                    ds-=d-strli_cez[i]
                    k+=1
                    d = abs(self.paths[i].r[k] - self.paths[i].r[k+1])
                    while  d <= ds: #preveri k index
                        ds -= d
                        k+=1
                        d = abs(self.paths[i].r[k] - self.paths[i].r[k+1])

                    strli_cez[i] = ds
                
                else:
                    strli_cez[i] += ds

                curent_idx[i] = k
                tmpR[i] = strli_cez[i] + self.paths[i].r[k]
            
            MAD = median_abs_deviation(tmpR)
            sigma2.append(MAD**2/0.45494)
        
        x = x[1:]
        x = np.log(np.array(x[20:]))
        sigma2 = np.log(np.array(sigma2[20:]))

        popt, pcov = curve_fit(lambda x,k,c: k*x + c, x, sigma2)
        return popt[0]

    def __init__(self, Npaths, path_len, μ, maksL, minL , seed=2718281828459045):
        self.μ = μ
        self.Npaths = Npaths
        self.path_len = path_len
        self.seed = seed
        self.b = maksL
        self.a = minL
        self.paths = []

class Path(Paths):
    a = 0.01 #smalest step
    b = 0.3 #biggest step

    def F(self, x):
        μ = self.μ
        return (pow(x,1-μ))/(1-μ)
    def invF(self, x):
        μ = self.μ
        return pow(x*(1-μ), 1/(1-μ))

    def get_l(self):
        a = self.a
        b = self.b
        p = self.random.random()
        # x =  p* self.F(self, b) + self.F(self, a) *(1-p)
        l = self.invF(p*self.F(b) + self.F(a)*(1-p))
        return l

    def generate(self):
        i = 0
        for i in range(self.N):
            phi = self.random.random() * np.pi * 2
            l = self.get_l()
            x = l * np.cos(phi)
            y = l * np.sin(phi)
            newx = self.x[-1] + x
            newy = self.y[-1] + y
            
            self.x.append(newx)
            self.y.append(newy)
            self.r.append(np.sqrt(newx**2 + newy**2))
    
    def add(self):
        phi = self.random.random() * np.pi * 2
        l = self.get_l()
        x = l * np.cos(phi)
        y = l * np.sin(phi)
        newx = self.x[-1] + x
        newy = self.y[-1] + y
        
        self.x.append(newx)
        self.y.append(newy)
        self.r.append(np.sqrt(newx**2 + newy**2))
        self.N += 1
    
    def draw(self, mix, max, miy, may):
        plt.plot(self.x, self.y,color="black",linewidth=0.2)
        plt.xlim(mix,max)
        plt.ylim(miy,may)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.plot(0,0, ".", color="red", label = "$(0,0)$")
        plt.legend()
        plt.title("$\mu = {}$, $N = {}$".format(self.μ, self.N))
        name = "mu={}.pdf".format(self.μ)
        plt.savefig(self.PATH+name, bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.clf()

    def __init__(self, N, μ, maksL, minL, seed=2718281828459045):
        self.N = N
        self.x, self.y, self.r = [0], [0], [0]
        self.μ = μ
        self.random = random
        self.random.seed(seed)
        self.b = maksL
        self.a = minL


