import math
import numpy as np

# Define variables
rho_gamma1 = 10
rho_gamma2 = 10 ** 3
rho_gamma3 = 10 ** 6
err = 1e-5
nx = 100
ny = 50
imax = nx + 2
jmax = ny + 2
xmax = 1
xmin = -1
ymax = 1
ymin = 0
dx = (xmax - xmin) / nx
dy = (ymax - ymin) / ny

# Create the mesh

# nodes positions
Px = np.arange(1, imax + 1)
Py = np.arange(1, jmax + 1)

# boundary conditions

# first node
Px[0] = 0
Py[0] = 0

# second node
Px[1] = Px[0] + dx / 2
Py[1] = Py[0] + dy / 2

# generic node
for i in range(2, imax - 2):
    Px[i] = Px[i - 1] + dx

for j in range(2, jmax - 2):
    Py[j] = Py[j - 1] + dy

# dislast node
Px[imax - 2] = Px[imax - 3] + dx
Py[jmax - 2] = Py[jmax - 3] + dy

# last node
Px[imax - 1] = Px[imax - 2] + dx / 2
Py[jmax - 1] = Py[jmax - 2] + dy / 2

Px = Px + xmin
Py = Py + ymin

X, Y = np.meshgrid(Px, Py)
alpha = 10
phi = np.zeros((jmax, imax))

for i in range(imax):
    for j in range(jmax):
        xP = X[j, i]
        if j == 1 and xP > xmin and xP < ymin:  # inlet boundary
            phi[j, i] = 1 + np.tanh(alpha * (2 * xP + 1))
        elif i == 1 or i == imax or j == jmax:  # elsewhere boundary
            phi[j, i] = 1 - np.tanh(alpha)

scheme = 'UDS'
rho_gamma_used = rho_gamma1

an = np.zeros(X.shape)
as_ = np.zeros(X.shape)
ae = np.zeros(X.shape)
aw = np.zeros(X.shape)
ap = np.zeros(X.shape)
bp = np.zeros(X.shape)

for i in range(imax):
    for j in range(jmax):
        xP = X[j, i]
        yP = Y[j, i]

        # Corners
        if (i == 1 and j == 1) or (i == 1 and j == jmax) or (i == imax and j == 1) or (i == imax and j == jmax):
            an[j, i] = 0
            as_[j, i] = 0
            ae[j, i] = 0
            aw[j, i] = 0
            ap[j, i] = 1
            bp[j, i] = 0

        # Inlet boundary
        elif j == 1 and xmin < xP <= ymin:
            an[j, i] = 0
            as_[j, i] = 0
            ae[j, i] = 0
            aw[j, i] = 0
            ap[j, i] = 1
            bp[j, i] = 1 + math.tanh(alpha * (2 * xP + 1))

        # Outlet boundary
        elif j == 1 and ymin < xP < xmax:
            an[j, i] = 1
            as_[j, i] = 0
            ae[j, i] = 0
            aw[j, i] = 0
            ap[j, i] = 1
            bp[j, i] = 0

        elif i == 1 or i == imax or j == jmax:
            an[j, i] = 0
            as_[j, i] = 0
            ae[j, i] = 0
            aw[j, i] = 0
            ap[j, i] = 1
            bp[j, i] = 1 - math.tanh(alpha)

        else:
            # at the faces
            nP = yP + (dy / 2)
            sP = yP - (dy / 2)
            eP = xP + (dy / 2)
            wP = xP - (dy / 2)

        Dn = dx / dy
        Ds = dx / dy
        De = dy / dx
        Dw = dy / dx
        D = [Dn, Ds, De, Dw]

        Fn = -rho_gamma_used * dx * (-2 * xP * (1 - nP * nP))
        Fs = rho_gamma_used * dx * (-2 * xP * (1 - sP * sP))
        Fe = -rho_gamma_used * dy * (-2 * yP * (1 - eP * eP))
        Fw = rho_gamma_used * dy * (-2 * yP * (1 - wP * eP))
        F = [Fn, Fs, Fe, Fw]

        # Peclet number
        P = [D[0]/abs(F[0]), D[1]/abs(F[1]), D[2]/abs(F[2]), D[3]/abs(F[3])]
        print(j)
        # Schemes (only UDS tacked in to account for now)
        if scheme == 'UDS':
            Ap = 1  # the scheme coefficient
        f2 = F.append(0)
        F = np.array(f2).astype(float)
        a = D*Ap+np.max(F)
        an[j, i] = a[0]
        as_[j, i] = a[1]
        ae[j, i] = a[2]
        aw[j, i] = a[3]
        ap[j, i] = np.sum(a)
        bp[j, i] = 0

