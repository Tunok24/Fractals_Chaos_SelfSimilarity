import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

"""
    Code for Lorenz Attractor.
"""
# interesting points: sigma=10, rho= 65, beta=12; sigma=15, rho=28, beta=10 <-- This last one is really good, try with end_time = 700, 1500
# Interesting param values: 10, 11, 8/3; 10, 20, 8/3;
# Hopf Bifurcation at sigma_c = beta + 1. If sigma > sigma_c then strange attractor.
# For rho < 1, there's one equillibrium point - which is the origin, the global attractor.

sigma = 15
rho = 28
beta = 10

x_0, y_0, z_0 = 0.5, 0.5, 0.5
x, y, z = [x_0], [y_0], [z_0]
x_new, y_new, z_new = x_0, y_0, z_0

start_time = 0
end_time = 1500
dt  = 0.01
time = np.arange(start_time, end_time, dt)

for t in time:
    dx_dt = sigma * (y_new - x_new)
    dy_dt = x_new * (rho - z_new) - y_new
    dz_dt = x_new * y_new - beta * z_new

    x_new += dx_dt * dt
    y_new += dy_dt * dt
    z_new += dz_dt * dt

    x.append(x_new)
    y.append(y_new)
    z.append(z_new)

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title("Lorenz Attractor")
ax.plot(x, y, z, linewidth=0.1, color='orange')
plt.savefig("./fig/Lorenz_attractor.png", dpi=300)
plt.show()



# Update function for animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    return line,

# Animate and show
# ani = FuncAnimation(fig, update, frames=len(x), interval=1, blit=True)
# plt.show()