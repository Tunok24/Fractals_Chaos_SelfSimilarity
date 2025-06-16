import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

"""
    Code for an Attractor created by accident.
"""
# Attractor Parameters
a = 1.0        # alpha
b = 2.5         # beta
c = 1.3         # gamma
d = 1.9         # mu

x_0, y_0, z_0 = 0.5, 0.5, 0.5
x, y, z = [x_0], [y_0], [z_0]
x_new, y_new, z_new = x_0, y_0, z_0

start_time = 0
end_time = 1000
dt = 0.01
time = np.arange(start_time, end_time, dt)

for t in time:
    dx_dt = a * x_new * (1 - y_new) - b * z_new
    dy_dt = -c * y_new * (1 - x_new * x_new)
    dz_dt = d * x_new

    x_new += dx_dt * dt
    y_new += dy_dt * dt
    z_new += dz_dt * dt

    x.append(x_new)
    y.append(y_new)
    z.append(z_new)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, linewidth=0.2, color='black')
ax.view_init(elev=20, azim=-0)
plt.savefig("./fig/accident_attractor1.png", dpi=300)
plt.show()
