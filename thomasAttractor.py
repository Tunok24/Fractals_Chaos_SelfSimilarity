import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

"""
    Code for Thomas Attractor.
"""
# Params
b = 0.198

x_0, y_0, z_0 = 0.91, 0.17, 0.53
x, y, z = [x_0], [y_0], [z_0]
x_new, y_new, z_new = x_0, y_0, z_0

start_time = 0
end_time = 10000
dt  = 0.01
time = np.arange(start_time, end_time, dt)

for t in time:
    dx_dt = np.sin(y_new) - b * x_new
    dy_dt = np.sin(z_new) - b * y_new
    dz_dt = np.sin(x_new) - b * z_new

    x_new += dx_dt * dt
    y_new += dy_dt * dt
    z_new += dz_dt * dt

    x.append(x_new)
    y.append(y_new)
    z.append(z_new)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, linewidth=0.2, color='orange')
plt.savefig("./fig/Thomas_attractor.png", dpi=300)
plt.show()

# Update function for animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    return line,

# Animate and show
# ani = FuncAnimation(fig, update, frames=len(x), interval=1, blit=True)
# plt.savefig("./fig/Thomas_attractor.png", dpi=300)
# plt.show()