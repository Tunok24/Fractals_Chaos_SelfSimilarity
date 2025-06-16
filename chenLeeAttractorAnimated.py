import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

"""
    Code for Burke-Shaw Attractor.
"""
# Attractor Parameters
# Interesting param set: 
# (a) a, b, d = 3.5, -10.0, -1.38
# (b) a, b, d = 2.5, -10.0, -0.38
# (c) a, b, d = 2.5, -14.0, -3.38
a, b, d = 3.5, -10.0, -1.38

x_0, y_0, z_0 = 0.5, 0.5, 0.5
x, y, z = [x_0], [y_0], [z_0]
x_new, y_new, z_new = x_0, y_0, z_0

start_time = 0
end_time = 10
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



# Plot setup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Make background black
dark_grey = '#111111'  # near-black but not quite. Other tryable #101010, #151515, #1a1a1a
fig.patch.set_facecolor(dark_grey)
ax.set_facecolor(dark_grey)
ax.axis('off')

# Glowing trace + bright moving point
trace, = ax.plot([], [], [], color='orange', linewidth=0.3, alpha=0.6)
point, = ax.plot([], [], [], 'o', color='red', markersize=4, alpha=1.0)

# Axis limits
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))

# Frames
frames = np.arange(0, len(x), 10)  # downsample
azim_angle = 0  # initial
elev_angle = 20

# Update function
def update(i):
    global azim_angle
    global elev_angle
    point.set_data([x[i]], [y[i]])
    point.set_3d_properties(z[i])
    trace.set_data(x[:i], y[:i])
    trace.set_3d_properties(z[:i])
    
    ax.view_init(elev=elev_angle, azim=azim_angle)
    azim_angle += 0.8  # slow rotation - 1/10th of a degree change
    elev_angle -= 0.3
    return point, trace

# Animate
ani = animation.FuncAnimation(fig, update, frames=frames, interval=5, blit=True)

# Save with ffmpeg (faster + smoother than pillow)
ani.save("./fig/ChenLeeAnimated.mp4", writer='ffmpeg', fps=20, dpi=200)

print("Saved as ./fig/ChenLeeAnimated.mp4")
