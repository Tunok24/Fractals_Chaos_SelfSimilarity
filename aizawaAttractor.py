import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm


"""
    Code for Aizawa Attractor.
"""
simulation_name = "AizawaAttractor"

# Aizawa Attractor Parameters
a = 0.95
b = 0.7
c = 0.6
d = 3.5
e = 0.25
f = 0.1

x_0, y_0, z_0 = 0.24, 0.17, 0.53
x, y, z = [x_0], [y_0], [z_0]
x_new, y_new, z_new = x_0, y_0, z_0

start_time = 0
end_time = 500
dt = 0.01
time = np.arange(start_time, end_time, dt)

for t in time:
    dx_dt = (z_new - b) * x_new - d * y_new
    dy_dt = d * x_new + (z_new - b) * y_new
    dz_dt = c + a * z_new - (z_new**3) / 3 \
            - (x_new**2 + y_new**2) * (1 + e * z_new) + f * z_new * (x_new**3)

    x_new += dx_dt * dt
    y_new += dy_dt * dt
    z_new += dz_dt * dt

    x.append(x_new)
    y.append(y_new)
    z.append(z_new)


if (True):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, linewidth=0.2, color='black')
    ax.view_init(elev=20, azim=-0)
    plt.savefig("./fig/Aizawa_attractor.png", dpi=300)
    plt.show()




#######################################
################# OR ##################