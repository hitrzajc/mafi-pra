import Methods
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

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

t_span = (0, 10)  # Time interval
size = 10000
t_eval = np.linspace(t_span[0], t_span[1], size)
y0 = [T_zacetna]          # Initial condition
rtol = 1e-7  # Relative tolerance
atol = 1e-11  # Absolute tolerance
# Solve the ODE
sol = solve_ivp(dydt, t_span, y0, method='RK45', t_eval=t_eval,rtol=rtol,atol=atol)

# Plot the solution
# plt.figure(figsize=(8, 6))
plt.plot(sol.t, sol.y[0], label="$T'(t) = -k (T(t)-T_{zun})$")
plt.title("Resitvi")
plt.xlabel("Cas $t$")
plt.ylabel("Temperatura $T$")
plt.grid(True)
# plt.legend()
# plt.savefig(PATH+"res.pdf")
# plt.clf()

def dydt1(t, y, k=0.1, T_zun =0, A = 10):
    return -k * (y - T_zun) + A * np.sin(2*np.pi * t)


sol = solve_ivp(dydt1, t_span, y0, method='RK45', t_eval=t_eval,rtol=rtol,atol=atol)

# Plot the solution
# plt.figure(figsize=(8, 6))
plt.plot(sol.t, sol.y[0], label="$T'(t) = -k (T(t)-T_{zun}) + A\sin(2\pi t)$")
# plt.title("Resitev: ")
# plt.xlabel("Cas $t$")
# plt.ylabel("Temperatura $T$")
plt.grid(True)
plt.legend()
plt.savefig(PATH+"ress.pdf")
plt.clf()
