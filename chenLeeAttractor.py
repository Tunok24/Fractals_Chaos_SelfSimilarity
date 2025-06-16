import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

"""
    Code for Burke-Shaw Attractor.
"""
# Attractor Parameters
# Interesting param set: 
# (a) a, b, d = 3.5, -10.0, -1.38
# (b) a, b, d = 2.5, -10.0, -0.38
# (c) a, b, d = 2.5, -14.0, -3.38
a, b, d = 2.5, -17.0, -3.38

x_0, y_0, z_0 = 0.5, 0.5, 0.5
x, y, z = [x_0], [y_0], [z_0]
x_new, y_new, z_new = x_0, y_0, z_0

start_time = 0
end_time = 300
dt = 0.001
time = np.arange(start_time, end_time, dt)

for t in time:
    dx_dt = a * x_new - y_new * z_new
    dy_dt = b * y_new + x_new * z_new
    dz_dt = d * z_new + (x_new * y_new)/3

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
plt.savefig("./fig/ChenLee_attractor.png", dpi=300)
plt.show()
