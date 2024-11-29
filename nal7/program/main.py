import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import ellipk

PATH = "../latex/pdfs/"
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0
methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
# Constants
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

# Solve using numerical integration
def solve_pendulum(theta0, theta_dot0, t_max, step_size, method="Radau"):
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, step_size)
    sol = solve_ivp(pendulum_eq, t_span, [theta0, theta_dot0], t_eval=t_eval, method=method)
    return sol.t, sol.y


# Compute analytical period using elliptic integral
def analytical_period(theta0):
    m = np.sin(theta0 / 2)**2
    return 4 / omega_0 * ellipk(m)

def analytical_omega(theta0):
    t0 = analytical_period(theta0)
    return 2*np.pi / t0

# Simulate and analyze results
t_max = 20  # Total simulation time (s)
step_size = 0.001  # Time step (s)

plt.figure(figsize=(8, 6))
    
# Solve the pendulum motion
for method in methods:
    t, y = solve_pendulum(theta0, theta_dot0, t_max, step_size, method=method)
    theta, theta_dot = y

    # # Plot results
    # plt.figure(figsize=(8, 6))
    # plt.plot(t, theta, label=r"$\theta(t)$ (numerical)")
    # plt.title("Numerical Solution for Pendulum Motion")
    # plt.xlabel("Time (s)")
    # plt.ylabel(r"$\theta$ (rad)")
    # plt.grid()
    # plt.legend()
    # plt.show()

    # Compute energy
    energy_values = energy(theta, theta_dot)

    plt.plot(t, energy_values, label=method)

plt.title(r"Energija v odvisnosti časa za: $\ddot{\theta} = -\omega_0^2 \sin\theta$")

plt.xlabel("Čas [s]")
plt.ylabel("Energija")
plt.grid()
plt.legend()
plt.savefig(PATH+"energije.pdf")
plt.clf()

t, y = solve_pendulum(theta0, theta_dot0, t_max, step_size, method="Radau")
theta, theta_dot = y
plt.figure(figsize=(8, 6))
plt.plot(t, theta,color="black",label=r"$\ddot{\theta} = -\omega_0^2\sin\theta$")
plt.plot(t,np.cos(t*omega_0),color="red", linestyle="--", label=r"$\ddot{\theta} = -\omega_0^2\theta$")
plt.title("Trajektorija nihala")
plt.xlabel("Čas [s]")
plt.ylabel(r"$\theta(t)$ [rad]")
plt.legend()
plt.grid()
# plt.legend()
plt.savefig(PATH + "trajektorija.pdf")
plt.clf()

# Plot phase portrait
# plot_phase_portrait(t, theta, theta_dot)
thetas = np.linspace(0,np.pi,200)
colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))
t_span = (0, t_max)
t_eval = np.arange(0, t_max, step_size)

####################################
# from tqdm import tqdm
# for i in tqdm(range(len(thetas))):
# # for theta0, color in tqdm(zip(thetas, colors)):
#     theta0 = thetas[i]
#     color = colors[-i]
#     sol = solve_ivp(pendulum_eq, t_span, [theta0, theta_dot0],t_eval=t_eval, method="Radau")
#     # print(len(sol.y))
#     t = sol.t
#     theta , theta_dot = sol.y
#     plt.plot(theta,theta_dot,color=color)



# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(analytical_omega(thetas)), vmax=max(analytical_omega(thetas))))
# cbar = plt.colorbar(sm)
# cbar.set_label("$\omega$ - prava kotna hitrost")
# plt.title(r"Fazni diagram nihala za različne $\theta_0$")
# plt.xlabel(r"$\theta$ [rad]")
# plt.ylabel(r"$\dot{\theta}$ [rad/s]")

# xticks = [-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
# xtick_labels = [r"$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", 
#                 r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]

# plt.xticks(xticks, xtick_labels)

# plt.savefig(PATH+"fazni_dig.pdf")
# plt.clf()
#######################################
thetas = np.linspace(0,2*np.pi,500)
plt.plot(thetas, analytical_omega(thetas), color="black",label=r"$\omega$")
plt.title("Dejanska kotna hitrost v odvisnosti od začetne lege nihala")
plt.xlabel(r"$\theta_0 [rad]$")
plt.ylabel("Kotna hitrost [rad]")
xticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
xtick_labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
plt.axhline(y=omega_0, color="red", linestyle="--", label=r"$\omega_0$")
plt.xticks(xticks, xtick_labels)
plt.legend()
plt.grid()
plt.savefig(PATH+"w(theta).pdf")

plt.clf()
