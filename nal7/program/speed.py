import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import ellipk

PATH = "../latex/pdfs/"
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0


g = 9.81  # gravitational acceleration (m/s^2)
L = 1.0   # pendulum length (m)
omega_0 = np.sqrt(g / L)  # natural frequency (rad/s)

# Initial conditions
theta0 = 1.0  # initial angle (rad)
theta_dot0 = 0.0  # initial angular velocity (rad/s)

# Energy function
def energy(theta, theta_dot):
    return (1 - np.cos(theta)) + theta_dot**2 / (2 * omega_0**2)

# Equation of motion
def pendulum_eq(t, y):
    theta, theta_dot = y
    dydt = [theta_dot, -omega_0**2 * np.sin(theta)]
    return dydt

def pendulum_eq1(theta):
    # theta, theta_dot = y
    
    return -np.sin(theta)

N = 5000
from diffeq_2 import *
from tqdm import tqdm

methods = [solve_ivp, verlet, pefrl]
t_max = 30
y = [[] for i in range(len(methods))]
x = []

step = 0.001
t_span = (0, t_max)
t_eval = [0]
for i in tqdm(range(3,N,10)):
    x.append(i)
    t_span = (0, t_span[1]+step)
    while i != len(t_eval):
        t_eval.append(t_eval[-1]+step)

    for j in range(len(methods)):
        method = methods[j]
        if method.__name__ == "solve_ivp":
            start = time.time()
            method(pendulum_eq,  y0=[theta0, theta_dot0], t_eval=t_eval, t_span=t_span, method="Radau")
            end = time.time()
        else:
            start = time.time()
            method(pendulum_eq1,  y0=[theta0, theta_dot0], t_eval=t_eval, t_span=t_span, method="Radau")
            end = time.time()
        
        t = end - start
        y[j].append(t)

for i in range(len(methods)):
    plt.plot(x,y[i],label=methods[i].__name__)
plt.xlabel("$N$ - Število vhodnih podatkov")
plt.ylabel("$t[s]$")
plt.title("Časovna odvisnost algoritmov")
plt.legend()
plt.grid()
plt.savefig(PATH+"time.pdf")
plt.yscale("log")
plt.savefig(PATH+"timelog.pdf")
