import os
import numpy as np
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm

"""
    Code for Rossler Attractor.
"""

simulation_name = "RosslerAttractor"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
a, b, c = 0.15, 0.05, 5.5
x_0, y_0, z_0 = 0.1, 0.0, 0.0
start_time, end_time, dt = 0, 1000, 0.01
time = np.arange(start_time, end_time, dt)

# Lists to store values
x, y, z = [x_0], [y_0], [z_0]
x_new, y_new, z_new = x_0, y_0, z_0

# Euler integration
for _ in time[1:]:
    dx_dt = - y_new - z_new
    dy_dt = x_new + a * y_new
    dz_dt = b + z_new * (x_new - c)

    x_new += dx_dt * dt
    y_new += dy_dt * dt
    z_new += dz_dt * dt

    x.append(x_new)
    y.append(y_new)
    z.append(z_new)

# Save as CSV
df = pd.DataFrame({'time': time, 'x': x, 'y': y, 'z': z})
df.to_csv("rossler_dataset.csv", index=False)

# Plot
if True:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, linewidth=0.2, color='black')
    ax.view_init(elev=20, azim=-0)
    plt.savefig("./fig/Rossler_attractor.png", dpi=300)
    plt.show()



#######################################
################# OR ##################

if (False):
    # Plot setup
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Make background black
    dark_grey = '#101010'  # near-black but not quite. Other tryable #101010, #151515, #1a1a1a
    fig.patch.set_facecolor(dark_grey)
    ax.set_facecolor(dark_grey)
    ax.axis('off')

    # Glowing trace + bright moving point
    trace, = ax.plot([], [], [], color='orange', linewidth=0.15, alpha=0.75)
    point, = ax.plot([], [], [], 'o', color='red', markersize=4, alpha=1.0)

    # Axis limits
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))

    # Frames
    frames = np.arange(0, len(x), 10)  # downsample
    azim_angle = 0  # initial
    elev_angle = 20

    # Precompute segments
    def make_segments(x, y, z):
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        return np.concatenate([points[:-1], points[1:]], axis=1)

    segments = make_segments(x, y, z)

    # Normalize time for color mapping
    norm = Normalize(vmin=0, vmax=len(segments))
    colors = plt.cm.plasma(norm(np.arange(len(segments))))  # You can change colormap

    # Create Line3DCollection (gradient)
    # Add a dummy segment to avoid auto_scale_xyz error
    dummy_seg = make_segments(x[:2], y[:2], z[:2])
    trace = Line3DCollection(dummy_seg, linewidth=0.6)
    trace.set_alpha(0.9)
    ax.add_collection3d(trace)

    # Animated point
    point, = ax.plot([], [], [], 'o', color='white', markersize=4)
    # Text elements (HUD)
    param_text = ax.text2D(-0.1, 0.7, f"{simulation_name}\n a, b, c = {a}, {b}, {c}\n dx/dt = - y - z\n dy/dt = x + ay\n dz/dt = b + z(x - c)",
                        transform=ax.transAxes, color='white', fontsize=9, family='monospace')

    time_text = ax.text2D(-0.1, 0.90, "", transform=ax.transAxes, color='white', fontsize=9, family='monospace')
    pos_text  = ax.text2D(-0.1, 0.87, "", transform=ax.transAxes, color='white', fontsize=9, family='monospace')


    # Update function
    def update(i):
        global azim_angle
        global elev_angle
        # Color-mapped trail
        trace.set_segments(segments[:i])
        trace.set_color(colors[:i])

        # Moving point
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties(z[i])

        t_val = start_time + i * dt  # Since you're downsampling by 10
        time_text.set_text(f"Time: {t_val:.2f}")
        pos_text.set_text(f"Pos: ({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f})")

        # Rotating view
        ax.view_init(elev=elev_angle, azim=azim_angle)
        azim_angle += 0.1
        elev_angle += 0.02
        return point, trace, time_text, pos_text

    # Animate
    print("Animating ...")
    # Create a temp folder to store frames
    frame_dir = "./frames"
    os.makedirs(frame_dir, exist_ok=True)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1, blit=True)
    fig._animation = ani  # <-- important magic line here
    # Save with ffmpeg (faster + smoother than pillow)
    print("Saving Animation ...")
    # Render frames manually and save each as PNG
    print("Rendering frames manually...")
    for i, frame in enumerate(tqdm(frames, desc="Rendering")):
        update(frame)
        filename = f"{frame_dir}/frame_{i:05d}.png"
        plt.savefig(filename, dpi=200, facecolor=fig.get_facecolor())


    os.system(f"ffmpeg -r 25 -i ./frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 ./fig/{simulation_name}.mp4")
    shutil.rmtree(frame_dir)

    print(f"Saved as ./fig/{simulation_name}.mp4")