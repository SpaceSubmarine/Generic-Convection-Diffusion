import numpy as np
import matplotlib.pyplot as plt

# Parallel Flow
# Input Data
v_in_x = 20  # inlet velocity (m/s) in x-axis
v_in_y =0 # inlet velocity (m/s) in x-axis
#v_out_x = v_in_x  # inlet velocity (m/s) in x-axis
#v_out_y = v_in_y  # inlet velocity (m/s) in x-axis
N = 40 # Number of control volumes in x-axis
M = N  # Number of control volumes in y-axis
dens = 1.225 # water density
max_iter = 2000  # maximum number of iterations in the gauss-seidel
max_difFer = 1e-6
time = 10  # sec
delta_t = 1  # number of divisions
time_step = np.linspace(0, time, delta_t)
MaxIter = 1e5
error = 1e-6

# EDS (Exponential-Difference-Scheme)
rho_gamma = 10  # relation for the Peclet number density/(difussion coefficient)
L = 1  # m of length
H = L   # height in meters
# Pe = rho_gamma * v_in_x * L  # Peclet number for Exponential-Difference Scheme
Pe = 0.05
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
vx = np.ones(np.shape(x)) * v_in_x
vy = np.ones(np.shape(y)) * v_in_y

# Velocity restrictions at boundary walls
vx[:, 0] = v_in_x
vx[:, -1] = v_in_x
vx[0, :] = 0
vx[-1, :] = 0
vy[:, 0] = v_in_y
vy[:, -1] = v_in_y
vy[0, :] = 0
vy[-1, :] = 0

# Velocity module
vp = np.zeros(np.shape(x))
for j in range(M + 2):
    for i in range(N + 2):
        vp[i, j] = np.sqrt(vx[i, j] ** 2 + vy[i, j] ** 2)

# =============================================================================
# Initial PHI
# This initial PHI is for the gauss-seidel solver, it needs a first value to iterate
PHI = np.zeros(np.shape(x))

for i in range(N + 2):
    for j in range(M + 2):
        PHI[i, j] = vp[i, j] * 1  # y[i, j]



# INITIALIZING COEFFICIENTS
ae = np.ones_like(M_fluid)
aw = np.ones_like(M_fluid)
as_ = np.ones_like(M_fluid)
an = np.ones_like(M_fluid)
ap = np.ones_like(M_fluid)
bP = np.zeros_like(M_fluid)
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



# BOUNDARY CONDITIONS for the inlet and outlet
# the velocity at the inlet and outlet are equal but at the walls is 0
# inlet
for i in range(N + 2):
    bP[i, 0] = vp[i, 0] * y[i, 0]
    bP[i, -1] = vp[i, -1] * y[i, -1]
an[:, 0] = 0
as_[:, 0] = 0
ae[:, 0] = 0
aw[:, 0] = 0
# outlet
an[:, -1] = 0
as_[:, -1] = 0
ae[:, -1] = 0


ap[0, :] = 1
ae[0, :] = 0
aw[0, :] = 0
as_[0, :] = 0
an[0, :] = 0
ap[-1, :] = 1
ae[-1, :] = 0
aw[-1, :] = 0
as_[-1, :] = 0
an[-1, :] = 0


# Difussion coefficient computed using the desity, inlet velocity and the peclet number
Gamma = (dens * v_in_x * L) / Pe
iter_list = []
dif_list = []
dif = 0.001
Iter = 1
bpt = bP
vxP = np.zeros_like(M_fluid)
vyP = np.zeros_like(M_fluid)
vP = np.zeros_like(M_fluid)

# PHI is the general variable (velocity in momentum equation and temperature in energy equation)
while Iter < max_iter and dif > max_difFer:
    PHI_old = PHI.copy()
    error_list = [0]
    vp_old = vp.copy()
    for j in range(1, N + 1):
        for i in range(1, M + 1):

            PHI[0, :] = 0
            PHI[-1, :] = 0
            PHI[:, 0] = vp[:,0]
            PHI[0, :] = vp[0, :]
            #PHI[:, -1] = v_in_x
            Gamma = (dens * v_in_x * x[i, j]) / Pe

            De[i, j + 1] = (dens * v_in_x * x[i, j+1]) / Pe * dx * dy / dPE
            Dw[i, j - 1] = (dens * v_in_x * x[i, j-1]) / Pe * dx * dy / dPW
            Dn[i + 1, j] = (dens * v_in_x * x[i+1, j]) / Pe * dx * dy / dPN
            Ds[i - 1, j] = (dens * v_in_x * x[i-1, j]) / Pe * dx * dy / dPS

            me[i, j] = vx[i, j + 1] * dens * dx * dy
            mw[i, j] = vx[i, j - 1] * dens * dx * dy
            mn[i, j] = vy[i + 1, j] * dens * dx * dy
            ms[i, j] = vy[i - 1, j] * dens * dx * dy

            ae[i, j] = De[i, j + 1] - (me[i, j] - abs(me[i, j])) / 2
            aw[i, j] = Dw[i, j - 1] + (mw[i, j] + abs(mw[i, j])) / 2
            an[i, j] = Dn[i+1, j] - (mn[i, j] - abs(mn[i, j])) / 2
            as_[i, j] = Ds[i-1, j] + (ms[i, j] + abs(ms[i, j])) / 2
            ap[i, j] = (ae[i, j] + aw[i, j] + as_[i, j] + an[i, j])
            bpt[i, j] = ((dens * (dx ** 3)) / delta_t) * PHI_old[i, j] + me[i, j] * (PHI[i, j + 1]) + \
                        mw[i, j] * (PHI[i, j - 1]) + mn[i, j] * (PHI[i + 1, j]) + ms[i, j] * (PHI[i - 1, j])

            PHI[i, j] = (an[i, j] * PHI_old[i + 1, j] + as_[i, j] * PHI_old[i - 1, j] +
                        ae[i, j] * PHI_old[i, j + 1] + aw[i, j] * PHI_old[i, j - 1] + bpt[i, j]) / ap[i, j]

            vxn = an[i, j] * ((PHI[i+1, j] + PHI[i, j]) / dPN)
            vxs = as_[i, j] * ((PHI[i, j] + PHI[i-1, j]) / dPS)
            vye = ae[i, j] * ((PHI[i, j+1] - PHI[i, j]) / dPE)
            vyw = aw[i, j] * ((PHI[i, j] - PHI[i, j - 1]) / dPW)
            vxP[i, j] = (vxn + vxs) / 2
            vyP[i, j] = (vye + vyw) / 2
            vP[i, j] = np.sqrt(vxP[i, j] ** 2 + vyP[i, j] ** 2)
            
            
    Iter += 1
    iter_list.append(Iter)
    differ = np.max(np.max(PHI_old - PHI))
    dif_list.append(differ)
    error_list.append(np.absolute(PHI_old[i, j] - PHI[i, j]))
    dif = np.max(np.array(error_list))
    print(Iter, dif)

# ================================================

plt.style.use("dark_background")
plt.title("VELOCITY FIELD")
plt.xlabel('x-axis(m)')
plt.ylabel('y-axis(m)')
plt.streamplot(x,  y, vxP, vyP,density=1,
                          linewidth=0.55, arrowsize=0.6, color=np.flipud(vxP), cmap="jet")  # color=vxP
plt.autoscale()
plt.show()


plt.title("VELOCITY FIELD")
plt.xlabel('x-axis(m)')
plt.ylabel('y-axis(m)')
plt.quiver(X, Y, vxP, vyP, color='w', alpha=0.7, scale=5000, width=0.005)
plt.show()

plt.imshow(np.flip(vP), cmap='inferno', interpolation="gaussian")
plt.title("VELOCITY FIELD")
plt.xlabel('x-axis(m)')
plt.ylabel('y-axis(m)')
#plt.colorbar()
plt.show()


plt.pcolormesh(X, Y, M_fluid, cmap='inferno')
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

plt.title("PHI Map")
plt.xlabel('x-axis(m)')
plt.ylabel('y-axis(m)')
img1 = plt.imshow(PHI, cmap="inferno", interpolation="gaussian")  # , interpolation="gaussian"
plt.axis(img1.get_extent())
plt.colorbar()
plt.show()


# REPORT SECTION============================================
print("Report------------------------------------")
print(" Iterations: ", Iter, " Error: ", "{:.7f}".format(differ))
print("Density of the fluid = ", dens, "kg/m3")
print("Height of the chanel H = ", H, "(m)")
print("Longitude of the chanel L = ", L, "(m)")
print("Inlet velocity v_in = ", v_in_x, "(m/s) in x-axis ")
print("Maximum number of iterations in the Gauss-Seidel Solver max_iter = ", max_iter)
print("Maximum error for convergence  max_difFer = ", max_difFer)

