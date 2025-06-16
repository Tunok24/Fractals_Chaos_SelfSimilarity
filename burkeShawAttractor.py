import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

"""
    Code for Burke-Shaw Attractor.
"""
# Attractor Parameters
# Interesting param set: 
# (a)
# (b)
s = 10.0
v = 4.272


x_0, y_0, z_0 = 0.2, 0.5, 0.7
x, y, z = [x_0], [y_0], [z_0]
x_new, y_new, z_new = x_0, y_0, z_0

start_time = 0
end_time = 1000
dt = 0.01
time = np.arange(start_time, end_time, dt)

for t in time:
    dx_dt = -s*(x_new + y_new)
    dy_dt = -y_new - s*x_new*z_new
    dz_dt = s*x_new*y_new + v

    x_new += dx_dt * dt
    y_new += dy_dt * dt
    z_new += dz_dt * dt

    x.append(x_new)
    y.append(y_new)
    z.append(z_new)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, linewidth=0.05, color='black')
ax.view_init(elev=20, azim=-0)
plt.savefig("./fig/BurkeShaw_attractor.png", dpi=300)
plt.show()
