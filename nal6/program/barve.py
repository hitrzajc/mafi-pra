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


def dydt1(t, y, k=0.1, T_zun =0, A = 10):
    return -k * (y - T_zun) + A * np.sin(2*np.pi * t)
def dydt(t, y, k=0.1, T_zun =0, A = 10):
    return -k * (y - T_zun)

###################################
rtol = 1e-7  # Relative tolerance
atol = 1e-11  # Absolute tolerance
k_values = np.linspace(0.1, 2.0, 2000)  # Vary k from 0.5 to 2.0
colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))  # Generate colors for each k

for k, color in zip(k_values, colors):
    sol = solve_ivp(dydt1, t_span, y0, method='DOP853', t_eval=t_eval,args=[k],rtol=rtol,atol=atol)
    plt.plot(sol.t, sol.y[0], color=color,linewidth=1)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=k_values.min(), vmax=k_values.max()))
sm.set_array([])
plt.colorbar(sm, label='parameter k')
plt.xlabel("Cas t")
plt.ylabel("Temperatura T")
plt.title("Resitev: $T'(t) = -k (T(t)-T_{zun}) + A\sin(2\pi t)$")
plt.savefig(PATH+"T1(t,k).pdf",dpi=500) #zanimivi krogci
plt.yscale("log")
plt.savefig(PATH+"T1(t,k)_log.pdf",dpi=500)
plt.clf()





################################
k_values = np.linspace(0.1, 2.0, 2000)  # Vary k from 0.5 to 2.0
colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))  # Generate colors for each k

for k, color in zip(k_values, colors):
    sol = solve_ivp(dydt, t_span, y0, method='DOP853', t_eval=t_eval,args=[k],rtol=rtol,atol=atol)
    plt.plot(sol.t, sol.y[0], color=color,linewidth=1)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=k_values.min(), vmax=k_values.max()))
sm.set_array([])
plt.colorbar(sm, label='parameter k')
plt.xlabel("Cas t")
plt.ylabel("Temperatura T")
plt.title("Resitev: $T'(t) = -k (T(t)-T_{zun})$")
plt.savefig(PATH+"T(t,k).pdf",dpi=500)
plt.yscale("log")
plt.savefig(PATH+"T(t,k)_log.pdf",dpi=500) #zanimivi krogci


##############################
# k_values = np.linspace(0, 10, 2000)  # Vary k from 0.5 to 2.0
# colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))  # Generate colors for each k

# for A, color in zip(k_values, colors):
#     sol = solve_ivp(dydt1, t_span, y0, method='DOP853', t_eval=t_eval,args=[0.1,A])
#     plt.plot(sol.t, sol.y[0], color=color,linewidth=1)

# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=k_values.min(), vmax=k_values.max()))
# sm.set_array([])
# plt.colorbar(sm, label='parameter $T_{zun}$')
# plt.xlabel("Cas t")
# plt.ylabel("Temperatura T")
# plt.title("$T'(t) = -k (T(t)-T_{zun}) + A\sin(2\pi t)$ metoda DOP853")
# plt.savefig(PATH+"T_zac.pdf")
# plt.yscale("log")
# plt.savefig(PATH+"T_zac_log.pdf")
# plt.clf()

##########################
# k_values = np.linspace(0, 10, 2000)  # Vary k from 0.5 to 2.0
# colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))  # Generate colors for each k

# for A, color in zip(k_values, colors):
#     sol = solve_ivp(dydt1, t_span, y0, method='LSODA', t_eval=t_eval,args=[0.1,A])
#     plt.plot(sol.t, sol.y[0], color=color,linewidth=1)

# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=k_values.min(), vmax=k_values.max()))
# sm.set_array([])
# plt.colorbar(sm, label='parameter $T_{zun}$')
# plt.xlabel("Cas t")
# plt.ylabel("Temperatura T")
# plt.title("$T'(t) = -k (T(t)-T_{zun}) + A\sin(2\pi t)$ metoda LSODA")
# plt.savefig(PATH+"T_zac_LSODA.pdf")
# plt.yscale("log")
# plt.savefig(PATH+"T_zac_log_LSODA.pdf")
# plt.clf()



##############################
# methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
# k_values = np.linspace(0, 10, 2000)  # Vary k from 0.5 to 2.0
# colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))  # Generate colors for each k
# for method in tqdm(methods):

#     for A, color in zip(k_values, colors):
#         sol = solve_ivp(dydt1, t_span, y0, method=method, t_eval=t_eval,args=[0.1,T_zacetna,A])
#         plt.plot(sol.t, sol.y[0], color=color,linewidth=1)

#     sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=k_values.min(), vmax=k_values.max()))
#     sm.set_array([])
#     plt.colorbar(sm, label='parameter $A$')
#     plt.xlabel("Cas t")
#     plt.ylabel("Temperatura T")
#     plt.title("$T'(t) = -k (T(t)-T_{zun}) + A\sin(2\pi t)$ metoda "+method)
#     plt.savefig(PATH+"A_{}.pdf".format(method))
#     plt.yscale("log")
#     plt.savefig(PATH+"A_{}_log.pdf".format(method))
#     plt.clf()
###########################
# k_values = np.linspace(0, 10, 2000)  # Vary k from 0.5 to 2.0
# colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))  # Generate colors for each k

# rtol = 1e-7  # Relative tolerance
# atol = 1e-11  # Absolute tolerance
# method = 'DOP853'
# for A, color in zip(k_values, colors):
#     sol = solve_ivp(dydt1, t_span, y0, method=method, t_eval=t_eval,rtol=rtol, atol=atol,args=[0.1,T_zacetna,A])
#     plt.plot(sol.t, sol.y[0], color=color,linewidth=1)

# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=k_values.min(), vmax=k_values.max()))
# sm.set_array([])
# plt.colorbar(sm, label='parameter $A$')
# plt.xlabel("Cas t")
# plt.ylabel("Temperatura T")
# plt.title("$T'(t) = -k (T(t)-T_{zun}) + A\sin(2\pi t)$ metoda "+method)
# plt.savefig(PATH+"A_tocna.pdf".format(method))
# plt.clf()