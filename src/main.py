import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# --- paramètres physiques ---
g = 9.81
mass_ball = 0.02
mu = 0.6
dt = 0.002
steps = 4000

# Soleil
M_sun = 10.0         # "masse" du soleil (unités arbitraires)
sun_radius = 0.3

# échelles pour transformer M_sun → profondeur / largeur du puits
k_depth = 0.03
k_sigma = 2.0

depth = k_depth * M_sun
sigma = k_sigma * sun_radius

# bille
x0, y0 = 1.0, 0.0
vx0, vy0 = 0.0, 0.6
ball_radius = 0.05

# domaine
grid_range = 2.0
n_grid = 120

# --- surface du drap ---
def h(x, y):
    r2 = x**2 + y**2
    return -depth * np.exp(-r2 / (2 * sigma**2))

def dh_dx(x, y):
    return -depth * np.exp(-(x**2 + y**2)/(2*sigma**2)) * (-x / sigma**2)

def dh_dy(x, y):
    return -depth * np.exp(-(x**2 + y**2)/(2*sigma**2)) * (-y / sigma**2)

def normal(x, y):
    nx = -dh_dx(x, y)
    ny = -dh_dy(x, y)
    nz = 1.0
    n = np.array([nx, ny, nz])
    return n / np.linalg.norm(n)

# --- simulation ---
def simulate():
    xs, ys, zs = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    x, y = x0, y0
    v = np.array([vx0, vy0, 0.0])
    n = normal(x, y)
    v -= n * np.dot(v, n)

    for i in range(steps):
        xs[i], ys[i], zs[i] = x, y, h(x, y)
        n = normal(x, y)
        g_vec = np.array([0, 0, -g])
        a_tangent = g_vec - n * np.dot(g_vec, n)
        v_tangent = v - n * np.dot(v, n)
        a_diss = - (mu / mass_ball) * v_tangent
        a = a_tangent + a_diss
        v += a * dt
        x += v[0] * dt
        y += v[1] * dt
        n2 = normal(x, y)
        v -= n2 * np.dot(v, n2)
    return xs, ys, zs

xs, ys, zs = simulate()

# --- affichage ---
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

lin = np.linspace(-grid_range, grid_range, n_grid)
X, Y = np.meshgrid(lin, lin)
Z = h(X, Y)
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=0, alpha=0.9)

# Soleil (centre)
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 25)
X_sun = sun_radius * np.outer(np.cos(u), np.sin(v))
Y_sun = sun_radius * np.outer(np.sin(u), np.sin(v))
Z_sun = sun_radius * np.outer(np.ones_like(u), np.cos(v)) + h(0,0)
ax.plot_surface(X_sun, Y_sun, Z_sun, color='gold', shade=True)

# Bille
ball = ax.plot([xs[0]], [ys[0]], [zs[0]+ball_radius], marker='o', color='blue')[0]

# Fixer échelle égale
def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_range, y_range, z_range = [abs(l[1]-l[0]) for l in (x_limits, y_limits, z_limits)]
    max_range = max(x_range, y_range, z_range)
    centers = [np.mean(l) for l in (x_limits, y_limits, z_limits)]
    ax.set_xlim3d([centers[0]-max_range/2, centers[0]+max_range/2])
    ax.set_ylim3d([centers[1]-max_range/2, centers[1]+max_range/2])
    ax.set_zlim3d([centers[2]-max_range/2, centers[2]+max_range/2])

def update(i):
    ball.set_data([xs[i]], [ys[i]])
    ball.set_3d_properties([zs[i] + ball_radius])
    return ball,

anim = animation.FuncAnimation(fig, update, frames=len(xs), interval=12, blit=True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("Bille sur un drap déformé par la masse du Soleil")
set_axes_equal(ax)

plt.show()
