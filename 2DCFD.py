import numpy as np
# import matplotlib.pyplot as plt
import math
# import seaborn as sns

########################################################################################################################
# Input-DATA
r = 3.2  # Radius of the cylinder (m)
v = 8  # Velocity (m/s)
dens = 1025  # density of sea water (kg/m3)
H = 10  # Height (m)
L = 20  # length (m)
tol = 0.4  # tolerance
vOut = v
max_iteration = 1e5
max_diff = 1e-6

########################################################################################################################
# Mesh
dx = tol
dy = tol
M = round(H / dy)
N = round(L / dx)
X = np.linspace(0, L, N + 2)
Y = np.linspace(H, 0, M + 2)[:, np.newaxis]
x, y = np.meshgrid(X, Y)


########################################################################################################################
# Mfluidity
Mfluid = np.ones(x.shape)
Mdens = np.ones(Mfluid.shape) * dens


########################################################################################################################
# Central Point of Radius
py = round((M + 2) / 2)
px = round((N + 2) / 2)

bP = np.zeros(Mfluid.shape)


########################################################################################################################
# Cylinder
for j in range(int(px - r / dx), int(px + r / dx) + 1):
    for i in range(int(py - r / dy), int(py + r / dy) + 1):
        if (np.sqrt((x[i, j] - x[py, px]) ** 2 + (y[i, j] - y[py, px]) ** 2)) < r:
            Mfluid[i, j] = 0
            bP[i, j] = v * H / 2


########################################################################################################################
# a_Coefficients
ae = np.ones(Mfluid.shape)
aw = np.ones(Mfluid.shape)
as_ = np.ones(Mfluid.shape)
an = np.ones(Mfluid.shape)
ap = np.ones(Mfluid.shape)


########################################################################################################################
# conditions
bP[:, 0] = v * Y
an[:, 0] = 0
as_[:, 0] = 0
ae[:, 0] = 0
aw[:, 0] = 0
an[:, -1] = 0
as_[:, -1] = 0
ae[:, -1] = 0
bP[0, :] = H * v

dPE = dx
dPW = dx
dPS = dy
dPN = dy
dPe = dx / 2
dPw = dx / 2
dPs = dy / 2
dPn = dy / 2
dEe = dx / 2
dWw = dx / 2
dSs = dy / 2
dNn = dy / 2


########################################################################################################################
Mdens *= Mfluid

# if(Mfluid == 1);
# Esto arrglea los coeficientes en el interior del cilindro
for j in range(int(px - r / dx), int(px + r / dx) + 1):
    for i in range(int(py - r / dy), int(py + r / dy) + 1):
        if (np.sqrt((x[i, j] - x[py, px]) ** 2 + (y[i, j] - y[py, px]) ** 2)) < r:
            ap[i, j] = 1
            ae[i, j] = 0
            aw[i, j] = 0
            as_[i, j] = 0
            an[i, j] = 0


########################################################################################################################
# GAUSS-SEIDEL METHOD
PSI = (y * v) * Mfluid
Mdens *= Mfluid
bP[:, -1] = vOut * Y
differ = np.inf
iteration = 0
PSI_old = PSI
tol = 0.0001

while differ > max_diff and iteration < max_iteration:
    PSI_old = PSI
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if Mfluid[i, j] == 1:
                ae[i, j] = ((dPE / ((dPe / (dens / Mdens[i, j])) + (dEe / (dens / Mdens[i, j + 1])))) * (dy / dPE))
                ind = np.where(np.isinf(ae))
                ae[ind] = 1
                ind = np.where(np.isnan(ae))
                ae[ind] = 1

                aw[i, j] = (dPW / ((dPw / (dens / Mdens[i, j])) + (dWw / (dens / Mdens[i, j - 1])))) * (dy / dPW)
                ind = np.where(np.isinf(aw))
                aw[ind] = 1

                as_[i, j] = (dPS / ((dPs / (dens / Mdens[i, j])) + (dSs / (dens / Mdens[i + 1, j])))) * (dx / dPS)
                ind = np.where(np.isinf(as_))
                as_[ind] = 1
                ind = np.where(np.isnan(as_))
                as_[ind] = 1

                an[i, j] = (dPN / ((dPn / (dens / Mdens[i, j])) + (dNn / (dens / Mdens[i - 1, j])))) * (dx / dPN)
                ind = np.where(np.isinf(an))
                an[ind] = 1
                ind = np.where(np.isnan(an))
                an[ind] = 1

                ap[i, j] = ae[i, j] + aw[i, j] + as_[i, j] + an[i, j]
                ind = np.where(np.isinf(ap))
                ap[ind] = 1
                ind = np.where(np.isnan(ap))
                ap[ind] = 1

            PSI[i, j] = (ae[i, j] * PSI[i, j + 1] +
                         aw[i, j] * PSI[i, j - 1] + an[i, j] * PSI[i - 1, j] +
                         as_[i, j] * PSI[i + 1, j] + bP[i, j]) / ap[i, j]
            iteration += 1
            differ = np.max(np.abs(PSI_old - PSI))

PSI[:, -1] = PSI[:, -1 - 1]


########################################################################################################################
# VELOCITY
vxP = v * Mfluid
vyP = np.zeros_like(Mfluid)
vP = v * Mfluid
vxn = np.ones_like(Mfluid)
vxs = np.ones_like(Mfluid)
vye = np.ones_like(Mfluid)
vyw = np.ones_like(Mfluid)

for i in range(2, M + 1):
    for j in range(2, N + 1):
        if Mfluid[i, j] == 1:
            vxn = an[i, j] * (PSI[i - 1, j] - PSI[i, j]) / dPN
            vxs = as_[i, j] * (PSI[i, j] - PSI[i + 1, j]) / dPS
            vye = -ae[i, j] * (PSI[i, j + 1] - PSI[i, j]) / dPE
            vyw = -aw[i, j] * (PSI[i, j] - PSI[i, j - 1]) / dPW

            vxP[i, j] = (vxn + vxs) / 2
            vyP[i, j] = (vye + vyw) / 2

            vP[i, j] = math.sqrt(vxP[i, j] ** 2 + vyP[i, j] ** 2)

            vP[i, j] = math.sqrt(vxP[i, j] ** 2 + vyP[i, j] ** 2)

PSI = PSI * Mfluid
