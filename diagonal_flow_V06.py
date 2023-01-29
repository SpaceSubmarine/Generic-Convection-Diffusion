import math
import numpy as np
import matplotlib.pyplot as plt

# Parallel Flow
# Input Data
v0 = 0.2  # inlet velocity (m/s) in x-axis

N = 50  # Number of control volumes in x-axis
M = N  # Number of control volumes in y-axis
dens = 1.225  # water density
max_iter = 20000  # maximum number of iterations in the gauss-seidel
max_difFer = 1e-6
time = 10  # sec
delta_t = 1  # number of divisions
time_step = np.linspace(0, time, delta_t)
MaxIter = 1e5
error = 1e-6
alpha = 45  # degrees

# EDS (Exponential-Difference-Scheme)
rho_gamma = 1  # relation for the Peclet number density/(difussion coefficient)
L = 1  # m of length
H = L  # height in meters
# Pe = rho_gamma * v_in_x * L  # Peclet number for Exponential-Difference Scheme
Pe = 0.01
# mesh ===========================================
xmax = 1
xmin = 0
ymax = 1
ymin = 0

dx = (xmax - xmin) / N
dy = (ymax - ymin) / M
X = np.linspace(xmin, xmax, N + 2)
Y = np.linspace(ymax, ymin, M + 2)[::-1]
x, y = np.meshgrid(X, Y)

# Fluid Region
M_fluid = np.ones(np.shape(x))

# Velocity Field
vx = np.ones(np.shape(x)) * v0 * math.cos(math.degrees(alpha))
vy = np.ones(np.shape(y)) * v0 * math.sin(math.degrees(alpha))
vp = np.ones_like(vx)

for i in range(M + 2):
    for j in range(N + 2):
        vp[i, j] = math.sqrt(vx[i, j] ** 2 + vy[i, j] ** 2)

# Velocity restrictions at boundary walls
vx[0, :] = 0
vx[-1, :] = 0
vy[0, :] = 0
vy[-1, :] = 0

# Initial PHI
# This initial PHI is for the gauss-seidel solver, it needs a first value to iterate
PHI = np.zeros(np.shape(x))

for i in range(N + 2):
    for j in range(M + 2):
        PHI[i, j] = vp[i, j]  # y[i, j]

PHI = x+y

# BOUNDARY CONDITIONS =======================================
# Left
PHI[:, -1] = v0
# Bottom
PHI[0, :] = v0
# Top
PHI[-1, :] = 0
# Right
PHI[:, 0] = 0

# INITIALIZING COEFFICIENTS
ae = np.ones_like(M_fluid)
aw = np.ones_like(M_fluid)
as_ = np.ones_like(M_fluid)
an = np.ones_like(M_fluid)
ap = np.ones_like(M_fluid)
bp = np.zeros_like(M_fluid)
me = np.zeros_like(M_fluid)
mw = np.zeros_like(M_fluid)
ms = np.zeros_like(M_fluid)
mn = np.zeros_like(M_fluid)
De = np.zeros_like(M_fluid)
Dw = np.zeros_like(M_fluid)
Ds = np.zeros_like(M_fluid)
Dn = np.zeros_like(M_fluid)

dPE = dx
dPW = dx
dPS = dy
dPN = dy

plt.style.use("dark_background")
plt.imshow(np.flip(-PHI), cmap='inferno', interpolation="gaussian")
plt.title("Initial PHI Map")
plt.xlabel('N')
plt.ylabel('M')
plt.show()

plt.title("Initial VELOCITY FIELD")
plt.xlabel('x-axis(m)')
plt.ylabel('y-axis(m)')
plt.quiver(x, y, vx, vy, color='w')
plt.show()


# GAUSS-SEIDEL SOLVER ===================================
iter_list = []
dif_list = []
dif = 0.1
Iter = 1


# PHI is the general variable (velocity in momentum equation and temperature in energy equation)
while Iter < max_iter and dif > max_difFer:
    PHI_old = PHI.copy()
    error_list = [0]
    vp_old = vp.copy()
    for j in range(1, N + 1):
        for i in range(1, M + 1):
            # Left
            De[i, j + 1] = -(dens * vp[i, j+1] * math.sqrt(x[i, j + 1]**2+y[i, j + 1]**2)) / Pe * dx * dy / dPE
            Dw[i, j - 1] = -(dens * vp[i, j-1] * math.sqrt(x[i, j - 1]**2+y[i, j - 1]**2)) / Pe * dx * dy / dPW
            Dn[i + 1, j] = -(dens * vp[i+1, j] * math.sqrt(x[i + 1, j]**2+y[i + 1, j]**2)) / Pe * dx * dy / dPN
            Ds[i - 1, j] = (dens * vp[i-1, j] * math.sqrt(x[i - 1, j]**2+y[i - 1, j]**2)) / Pe * dx * dy / dPS

            me[i, j] = vx[i, j + 1] * dens * dx * dy
            mw[i, j] = vx[i, j - 1] * dens * dx * dy
            mn[i, j] = vy[i + 1, j] * dens * dx * dy
            ms[i, j] = vy[i - 1, j] * dens * dx * dy

            ae[i, j] = De[i, j] + (me[i, j] + abs(me[i, j])) / 2
            ind = np.isinf(ae)
            ae[ind] = 1
            ind = np.isnan(ae)
            ae[ind] = 1
            aw[i, j] = -Dw[i, j] + (mw[i, j] + abs(mw[i, j])) / 2
            ind = np.isinf(aw)
            aw[ind] = 1
            ind = np.isnan(aw)
            aw[ind] = 1
            an[i, j] = Dn[i, j] + (mn[i, j] + abs(mn[i, j])) / 2
            ind = np.isinf(an)
            an[ind] = 1
            ind = np.isnan(an)
            an[ind] = 1
            as_[i, j] = -Ds[i, j] + (ms[i, j] + abs(ms[i, j])) / 2
            ind = np.isinf(as_)
            as_[ind] = 1
            ind = np.isnan(as_)
            as_[ind] = 1
            ap[i, j] = -(ae[i, j] + aw[i, j] + as_[i, j] + an[i, j])

            bp[i, j] = ((dens * (dx ** 3)) / delta_t) * PHI_old[i, j] + me[i, j] * (PHI[i, j + 1]) + \
                    mw[i, j] * (PHI[i, j - 1]) - mn[i, j] * (PHI[i + 1, j]) - ms[i, j] * (PHI[i - 1, j])

            PHI[i, j] = (an[i, j] * PHI_old[i + 1, j] + as_[i, j] * PHI_old[i - 1, j] +
                        ae[i, j] * PHI_old[i, j + 1] + aw[i, j] * PHI_old[i, j - 1] + bp[i, j]) / ap[i, j]

            vxn = -an[i, j] * ((PHI[i + 1, j] - PHI[i, j]) / dPN)
            vxs = -as_[i, j] * ((PHI[i, j] - PHI[i - 1, j]) / dPS)
            vye = ae[i, j] * ((PHI[i, j + 1] - PHI[i, j]) / dPE)
            vyw = -aw[i, j] * ((PHI[i, j] - PHI[i, j - 1]) / dPW)
            vx[i, j] =- (vxn + vxs) / 2
            vy[i, j] = (vye + vyw) / 2
            vp[i, j] = np.sqrt(vx[i, j] ** 2 + vy[i, j] ** 2)

    Iter += 1
    iter_list.append(Iter)
    differ = np.max(np.max(PHI_old - PHI))
    dif_list.append(differ)
    error_list.append(np.absolute(PHI_old[i, j] - PHI[i, j]))
    dif = np.max(np.array(error_list))
    print(Iter, dif)


#  PLOT SECTION =============================================
plt.imshow(PHI, cmap="inferno", interpolation='gaussian')
plt.title("PHI Map")
plt.xlabel('N')
plt.ylabel('M')
plt.colorbar()
plt.show()

plt.title("VELOCITY FIELD")
plt.xlabel('x-axis(m)')
plt.ylabel('y-axis(m)')
plt.quiver(x, y, np.flipud(-vx), np.flipud(vy), color='w')
plt.show()


'''plt.pcolormesh(X, Y, M_fluid, cmap='inferno')
plt.xticks(X)
plt.yticks(Y)
plt.grid(True, c="white", lw=0.5)
# plt.scatter(X, Y, color='red')
plt.colorbar()
plt.title("Mesh with N = " + str(N) + " divisions in x, M = " + str(M) + " divisions in y")
plt.show()


# coefficients visualization
plt.imshow(ap)
plt.show()
plt.imshow(ae)
plt.show()
plt.imshow(as_)
plt.show()
plt.imshow(an)
plt.show()
plt.imshow(aw)
plt.show()
'''


