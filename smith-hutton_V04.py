import time
import numpy as np
import matplotlib.pyplot as plt

# Clear command history
tic = time.time()

# Input Data
N = 50
M = N
alpha = 5

# UDS
Ape = 1
rho_gamma = 15
error = 1e-6
MaxIter = 1e5

# mesh
xmax = 1
xmin = -1
ymax = 1
ymin = 0
xmid = 0
dx = (xmax - xmin) / N
dy = (ymax - ymin) / M
X = np.linspace(xmin, xmax, N + 2)
Y = np.linspace(ymax, ymin, M + 2)[::-1]
x, y = np.meshgrid(X, Y)

# Node Position
Px = np.arange(1, N + 3)
Py = np.arange(1, M + 3)

# first node
Px[0] = 0
Py[0] = 0

# second node
Px[1] = Px[0] + dx / 2
Py[1] = Py[0] + dy / 2

# generic node
for i in range(2, N + 1):
    Px[i] = Px[i - 1] + dx

for j in range(2, M + 1):
    Py[j] = Py[j - 1] + dy

# dislast node
Px[N] = Px[N - 1] + dx
Py[M] = Py[M - 1] + dy

# last node
Px[N + 1] = Px[N] + dx / 2
Py[M + 1] = Py[M] + dy / 2

for i in range(N + 2):
    Px[i] = Px[i] + xmin

for j in range(M + 2):
    Py[j] = Py[j] + ymin

PX, PY = np.meshgrid(Px, Py)

# Fluid Region
M_fluid = np.ones(np.shape(x))

# Velocity Field
vx = np.ones(np.shape(x))
vy = np.ones(np.shape(y))

for j in range(M + 2):
    for i in range(N + 2):
        vx[i, j] = 2 * y[i, j] * (1 - x[i, j] ** 2)
        vy[i, j] = -2 * x[i, j] * (1 - y[i, j] ** 2)

# Velocity restrictions at boundary walls
vx[:, 0] = 0
vx[:, -1] = 0
vx[0, :] = 0
vy[:, 0] = 0
vy[:, -1] = 0
vy[0, :] = 0

# Velocity module
vp = np.zeros(np.shape(x))
for j in range(M + 2):
    for i in range(N + 2):
        vp[i, j] = np.sqrt(vx[i, j] ** 2 + vy[i, j] ** 2)

# Initial PHI
PHI = np.zeros(np.shape(x))
for i in range(N + 2):
    for j in range(M + 2):
        xP = x[i, j]
        if i == 1 and xmin < xP < xmid:  # inlet boundary
            PHI[i, j] = 1 + np.tanh(alpha * (2 * xP + 1))
        elif j == 1 or j == (N + 2) or i == (M + 2):  # elsewhere boundary
            PHI[i, j] = 1 - np.tanh(alpha)

phi_else = 1 - np.tanh(alpha)

# Coeficients
an = np.zeros((M + 2, N + 2))
as_ = np.zeros((M + 2, N + 2))
ae = np.zeros((M + 2, N + 2))
aw = np.zeros((M + 2, N + 2))
ap = np.zeros((M + 2, N + 2))
bp = np.zeros((M + 2, N + 2))

for i in range(N + 1):
    for j in range(M + 1):

        xP = x[i, j]
        yP = y[i, j]

        # corners definition
        if i == 0 and j == 0:
            an[i, j] = 0
            as_[i, j] = 0
            ae[i, j] = 0
            aw[i, j] = 0
            ap[i, j] = 1
            bp[i, j] = 0
        elif i == 0 and j == M + 2:
            an[i, j] = 0
            as_[i, j] = 0
            ae[i, j] = 0
            aw[i, j] = 0
            ap[i, j] = 1
            bp[i, j] = 0
        elif i == N + 2 and j == 0:
            an[i, j] = 0
            as_[i, j] = 0
            ae[i, j] = 0
            aw[i, j] = 0
            ap[i, j] = 1
            bp[i, j] = 0
        elif i == N + 2 and j == M + 2:
            an[i, j] = 0
            as_[i, j] = 0
            ae[i, j] = 0
            aw[i, j] = 0
            ap[i, j] = 1
            bp[i, j] = 0

            # Walls
            # bot_inlet
        elif i == 1 and xmin < xP <= xmid:
            an[i, j] = 0
            as_[i, j] = 0
            ae[i, j] = 0
            aw[i, j] = 0
            ap[i, j] = 1
            bp[i, j] = 1 + np.tanh(alpha * (2 * xP + 1))
            # bot_outlet
        elif i == 1 and xmid < xP < xmax:
            an[i, j] = 1
            as_[i, j] = 0
            ae[i, j] = 0
            aw[i, j] = 0
            ap[i, j] = 1
            bp[i, j] = 0
            # left,top,right Walls
        elif j == 1 or j == (M + 2) or i == (N + 2):
            an[i, j] = 0
            as_[i, j] = 0
            ae[i, j] = 0
            aw[i, j] = 0
            ap[i, j] = 1
            bp[i, j] = 1 - np.tanh(alpha)
        else:
            nP = yP + (dy / 2)
            sP = yP - (dy / 2)
            eP = xP + (dx / 2)
            wP = xP - (dx / 2)
            # D-coefficient:
            Dn = dx / dy
            Ds = dx / dy
            De = dy / dx
            Dw = dy / dx
            D = [Dn, Ds, De, Dw]
            Fn = -rho_gamma * dx * vy[i,j+1]#(2 * xP * (1 - nP * nP))
            Fs = rho_gamma * dx * vy[i,j-1]#(2 * xP * (1 - sP * sP))
            Fe = -rho_gamma * dy * vx[i+1,j]#(2 * yP * (1 - eP * eP))
            Fw = rho_gamma * dy * vx[i-1,j]#(2 * yP * (1 - wP * wP))
            F = [Fn, Fs, Fe, Fw]

            # Peclet number (P):
            P = np.absolute(np.array(F)) / np.array(D)
            scheme_coeff = Ape

            a = D * scheme_coeff + np.maximum(F, 0)
            an[i, j] = Dn * Ape + np.maximum(Fn, 0)
            as_[i, j] = Ds * Ape + np.maximum(Fs, 0)
            ae[i, j] = De * Ape + np.maximum(Fe, 0)
            aw[i, j] = Dw * Ape + np.maximum(Fw, 0)
            ap[i, j] =  (an[i, j] + as_[i, j] + ae[i, j] + aw[i, j])
            bp[i, j] = 1 - np.tanh(alpha)

bool = False
dif = 10
PHIold = PHI
Jmid = (M + 2) // 2
Jend = (M + 2)
Iter = 1
bp0 = bp
bpt = bp0 * PHI
while Iter < MaxIter and dif > error:
    PHIold = PHI.copy()
    error_list =[]
    for j in range(1, N + 1):
        for i in range(1, M + 1):
            ######################################################################
            #######################################################################
            xp = x[i, j]
            if i == 1 and xmin <= xp <= xmid:  # bot_inlet
                PHI[i, j] = 1 + np.tanh(alpha * (2 * xp + 1))

            elif j == 1 or j == M + 2 or i == (N + 2):  # Boundary Walls
                PHI[i, j] = phi_else

            elif i == 1 and xmid <= xp <= xmax:  # outlet boundary
                #PHI[i, j] = (an[i, j] * PHIold[i + 1, j] + ae[i, j] * PHIold[i, j + 1] +
                #        aw[i, j] * PHIold[i, j - 1] + bpt[i, j]) / ap[i, j]
                PHI[i, j] = 1 - np.tanh(alpha)
            else:
                PHI[i, j] = (an[i, j] * PHIold[i + 1, j] + as_[i, j] * PHIold[i - 1, j] +
                        ae[i, j] * PHIold[i, j + 1] + aw[i, j] * PHIold[i, j - 1] + bpt[i, j]) / ap[i, j]

            error_list.append(np.absolute(PHIold[i, j]-PHI[i, j]))
    Iter += 1
    dif = np.max(np.array(error_list))
    print(Iter, dif)


# PLOT
plt.style.use("dark_background")
plt.imshow(PHI, interpolation="gaussian", cmap="inferno", origin='lower')
plt.show()

figure1 = plt.figure(figsize=(10, 8), dpi=80)
plt.set_cmap('jet')
plt.contour(PHI, levels=64, colors="green", linewidths=0.5, antialiased=True)
plt.ylabel('Py Nodes')
plt.xlabel('Px Nodes')
plt.title(f'Distribution of \phi with {N + 2}x{M + 2} and scheme USD. rho/Gamma = {rho_gamma}')
plt.box(on=None)
plt.axis('tight')
plt.axis('equal')

plt.show()
