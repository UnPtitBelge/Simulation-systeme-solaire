import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# --- paramètres ---
G = 1.0          # constante gravitationnelle arbitraire
M_sun = 5.0      # masse du Soleil
mu = 0.05        # friction
dt = 0.01
steps = 4000

# Soleil et drap
sun_radius = 0.3
k_depth = 0.1
k_sigma = 2.0
depth = k_depth * M_sun
sigma = k_sigma * sun_radius

# --- drap ---
def h(x, y):
    r2 = x**2 + y**2
    return -depth * np.exp(-r2 / (2*sigma**2))

# --- simulation gravitationnelle ---
x, y = 2.0, 0.0
vx, vy = 0.0, 1.0
xs, ys, zs = [], [], []

for i in range(steps):
    r = np.sqrt(x**2 + y**2)
    if r < sun_radius:  # contact avec le Soleil
        break
    ax = -G * M_sun * x / r**3 - mu * vx
    ay = -G * M_sun * y / r**3 - mu * vy
    vx += ax * dt
    vy += ay * dt
    x += vx * dt
    y += vy * dt
    xs.append(x)
    ys.append(y)
    zs.append(h(x, y))

# --- affichage 3D ---
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

grid = np.linspace(-3, 3, 150)
X, Y = np.meshgrid(grid, grid)
Z = h(X, Y)
ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)

# Soleil
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 20)
X_sun = sun_radius * np.outer(np.cos(u), np.sin(v))
Y_sun = sun_radius * np.outer(np.sin(u), np.sin(v))
Z_sun = sun_radius * np.outer(np.ones_like(u), np.cos(v)) + h(0, 0)
ax.plot_surface(X_sun, Y_sun, Z_sun, color='gold', shade=True)

# bille animée
ball, = ax.plot([xs[0]], [ys[0]], [zs[0]+0.05], 'o', color='blue')

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
    ball.set_3d_properties([zs[i] + 0.05])
    return ball,

ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=10, blit=True)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title("Bille orbitant et tombant vers le Soleil")
set_axes_equal(ax)
plt.show()
