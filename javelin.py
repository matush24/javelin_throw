import numpy as np
import matplotlib.pyplot as plt

# set values
l0 = 2.6                            # length of javelin
C = 1.1                             # drag coefficient
rhov = 1.27                         # air density
g = np.array([0, 0, -9.81])         # gravitational acceleration
vabs = 35                           # throw speed
phi0 = 40                           # throw angle
pos0 = np.array([0, 0, 1.8])        # begining position

# setting numpy arrays
pos = np.array([pos0])
v = np.array([[vabs*np.cos(phi0/180*np.pi), 0, vabs*np.sin(phi0/180*np.pi)]])
phi = np.array([phi0/180*np.pi])
omega = np.array([0])

# diameter function
def h(l):
    return np.sin(np.pi*l/l0)/50 + 0.01

# density function
def rho(l):
    if l > 0.88*l0:
        return 6000
    else:
        return 500

# integration parameters
n = 100                             # number of parts to integrate over
dl = l0/n                           # length of one part
dt = 0.01                           # iteration time step

# calculate mass
m = 0
for l in np.linspace(0, l0, n):
    m += np.pi/4 * h(l)**2 * rho(l)*dl

# calculate center of mass
T = 0
for l in np.linspace(0, l0, n):
    T += np.pi/(4*m) * h(l)**2 * rho(l) * l * dl

# calculate moment of inertia
I = 0
for l in np.linspace(0, l0, n):
    I += np.pi/4 * h(l)**2 * rho(l) * dl * (l-T)**2

print(np.round(m, 3), np.round(l0- T, 3), np.round(I, 3))                  # print mass, distance of center of mass from the tip, moment of inertia

# integrate air resistance
def Fo():
    F = 0
    dlv = np.array([dl*np.cos(phi[-1]), 0, dl*np.sin(phi[-1])])                         # orientation of a single part
    alpha = np.arccos(max(np.dot(-v[-1], dlv)/(np.linalg.norm(-v[-1]) * dl), - 1))      # angle between orientation of the spear and wind speed
    for l in np.linspace(0, l0, n):
        Sk = np.sin(alpha)*dl*h(l)                                                      # efective area of a single part
        F += (1/2)*C*rhov*Sk*np.linalg.norm(v[-1])*v[-1]
    return F

# integrate moment of air resistance
def M():
    M = 0
    dlv = np.array([dl*np.cos(phi[-1]), 0, dl*np.sin(phi[-1])])                         # orientation of a single part
    alpha = np.arccos(max(np.dot(-v[-1], dlv)/(np.linalg.norm(-v[-1]) * dl), - 1))      # angle between orientation of the spear and wind speed
    for l in np.linspace(0, l0, n):
        posdl = np.array([(l-T)*np.cos(phi[-1]), 0, (l-T)*np.sin(phi[-1])])             # position of a integration part relative to the center of mass
        Sk = np.sin(alpha)*dl*h(l)                                                      # efective area of a single part
        M += np.cross(posdl, (1/2)*C*rhov*Sk*np.linalg.norm(v[-1])*v[-1])
    return M

# Compute movement using euler method
while pos[-1][2] >= 0:
    a = (m*g - Fo())/m
    pos = np.append(pos, [(pos[-1] + v[-1]*dt)], axis=0)
    v = np.append(v, [(v[-1] + a*dt)], axis=0)
    eps = M()[1]/I
    phi = np.append(phi, (phi[-1] + omega[-1]*dt))
    omega = np.append(omega, (omega[-1] + eps*dt)) 

# Prepare lists of positions for chart
x, y, z = [], [], []
for i in pos:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])

# Create chart (trajectory)
ax = plt.axes(projection='3d')
ax.plot(x, y, z)

# Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# show chart
plt.show()

# plot angle
plt.plot(x, phi)
plt.show()