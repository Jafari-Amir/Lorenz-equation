import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
# how conssider the Lorenz equation
def lorenz_equations(x, y, z, sigma, b, r):
    dx_dt = sigma * (y - x)
    dy_dt = x * (r - z) - y
    dz_dt = x * y - b * z
    return dx_dt, dy_dt, dz_dt

# parameters & initial condition(s)
sigma = 10
b = 8/3
r = 28
x0, y0, z0 = 10, 0, 0
t0, tf = 0, 25
dt = 0.011
N = int((tf - t0) / dt) + 1
# offer a solusion the Lorenz equations using the fourth-order Runge-Kutta method
t = np.linspace(t0, tf, N)
x, y, z = np.zeros(N), np.zeros(N), np.zeros(N)
x[0], y[0], z[0] = x0, y0, z0

for i in range(1, N):
    k1x, k1y, k1z = lorenz_equations(x[i-1], y[i-1], z[i-1], sigma, b, r)
    k2x, k2y, k2z = lorenz_equations(x[i-1] + 0.5*dt*k1x, y[i-1] + 0.5*dt*k1y, z[i-1] + 0.5*dt*k1z, sigma, b, r)
    k3x, k3y, k3z = lorenz_equations(x[i-1] + 0.5*dt*k2x, y[i-1] + 0.5*dt*k2y, z[i-1] + 0.5*dt*k2z, sigma, b, r)
    k4x, k4y, k4z = lorenz_equations(x[i-1] + dt*k3x, y[i-1] + dt*k3y, z[i-1] + dt*k3z, sigma, b, r)
    x[i] = x[i-1] + (1/6)*dt*(k1x + 2*k2x + 2*k3x + k4x)
    y[i] = y[i-1] + (1/6)*dt*(k1y + 2*k2y + 2*k3y+ k4y)
    z[i] = z[i-1] + (1/6)*dt*(k1z + 2*k2z + 2*k3z + k4z)
 
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
c = z
ax.scatter(x, y, z, c = c, edgecolor='black', cmap='plasma', linewidth=0.2,  s=30)
 # syntax for plotting
ax.set_title('3d Scatter plot of Lorenz equation solution')
plt.savefig('plot.png')
plt.show()