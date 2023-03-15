import numpy as np
import matplotlib.pyplot as plt

# iteration time step
dt = 0.01

# set values
lc = 1
C = 1
rhov = 1.27
g = np.array([0, 0, -9.81])

pos = np.array([[0, 0, 0]])
v = np.array([[30, 0, 30]])

phi = np.array([np.pi/4])
omega = np.array([0])

m = 1
start, stop = 0, 5

def h(l):
    return np.sin(np.pi*l/lc)/25

def rho(l):
    if l > 0.85*lc:
        return 7874
    else:
        return 900

n = 100
dl = lc/n

# calculate mass
m = 0
for l in np.linspace(0, lc, n):
    m += np.pi/4 * h(l)**2 * rho(l)*dl

# calculate center of mass
T = 0
for l in np.linspace(0, lc, n):
    T += np.pi/(4*m) * h(l)**2 * rho(l) * l * dl

# calculate moment of inertia
I = 0
for l in np.linspace(0, lc, n):
    I += np.pi/4 * h(l)**2 * rho(l) * dl * (l-T)**2

def Fo():
    F = 0
    for l in np.linspace(0, lc, n):
        dlv = np.array([dl*np.cos(phi[-1]), 0, dl*np.sin(phi[-1])])
        Sk = np.linalg.norm(dlv - np.dot(v[-1], dlv)*dlv/np.linalg.norm(v[-1]))*dl
        # dl*2*h(l)*(v[-1][0]/np.linalg.norm(v[-1])*np.sin(phi[-1]) - v[-1][1]/np.linalg.norm(v[-1])*np.cos(phi[-1]))
        F += (1/2)*C*rhov*Sk*np.linalg.norm(v[-1])*v[-1]
    return F

def M():
    M = 0
    for l in np.linspace(0, lc, n):
        posdl = np.array([(l-T)*np.cos(phi[-1]), 0, (l-T)*np.sin(phi[-1])])
        dlv = np.array([dl*np.cos(phi[-1]), 0, dl*np.sin(phi[-1])])
        Sk = np.linalg.norm(dlv - np.dot(v[-1], dlv)*dlv/np.linalg.norm(v[-1]))*dl
        M += np.cross(posdl, (1/2)*C*rhov*Sk*np.linalg.norm(v[-1])*v[-1])
    return M



# create time stamps
t = np.arange(start, stop, dt)

# Compute movement using euler method
while pos[-1][2] >= 0:
    a = (m*g - Fo())/m
    pos = np.append(pos, [(pos[-1] + v[-1]*dt)], axis=0)
    v = np.append(v, [(v[-1] + a*dt)], axis=0)
    eps = M()[1]/I
    phi = np.append(phi, (phi[-1] + omega[-1]*dt))
    omega = np.append(omega, (omega[-1] + eps*dt))
    # print(pos)
    

# Prepare lists of positions for chart
x, y, z = [], [], []
for i in pos:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])

# Create chart
ax = plt.axes(projection='3d')
ax.plot(x, y, z)

# Set labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# show chart
plt.show()

plt.plot(x, phi)
plt.show()