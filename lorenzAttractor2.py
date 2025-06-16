import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Lorenz parameters
# Interesting param values: 10, 11, 8/3; 10, 20, 8/3;
# Hopf Bifurcation at sigma_c = beta + 1. If sigma > sigma_c then strange attractor.
# For rho < 1, there's one equillibrium point - which is the origin, the global attractor.
sigma = 15
rho = 28
beta = 10

# Simulation settings
dt = 0.01
end_time = 1500
time = np.arange(0, end_time, dt)

# Initial conditions for both
x1_0, y1_0, z1_0 = 0.5, 0.5, 0.5
x2_0, y2_0, z2_0 = 0.51, 0.5, 0.5
x1, y1, z1 = [x1_0], [y1_0], [z1_0]       # Original
x2, y2, z2 = [x2_0], [y2_0], [z2_0]      # Slightly perturbed

# Simulate both
for _ in time:
    # First trajectory
    dx1 = sigma * (y1[-1] - x1[-1])
    dy1 = x1[-1] * (rho - z1[-1]) - y1[-1]
    dz1 = x1[-1] * y1[-1] - beta * z1[-1]
    x1.append(x1[-1] + dx1 * dt)
    y1.append(y1[-1] + dy1 * dt)
    z1.append(z1[-1] + dz1 * dt)

    # Second trajectory
    dx2 = sigma * (y2[-1] - x2[-1])
    dy2 = x2[-1] * (rho - z2[-1]) - y2[-1]
    dz2 = x2[-1] * y2[-1] - beta * z2[-1]
    x2.append(x2[-1] + dx2 * dt)
    y2.append(y2[-1] + dy2 * dt)
    z2.append(z2[-1] + dz2 * dt)

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x1, y1, z1, linewidth=0.2, color="orange")
ax.plot(x2, y2, z2, linewidth=0.2, color="red")
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_title("Lorenz Attractor — Diverging Trajectories")

# line1, = ax.plot([], [], [], color='blue', label='Initial: (0.5, 0.5, 0.5)')
# line2, = ax.plot([], [], [], color='red', label='Perturbed: (0.51, 0.5, 0.5)')
ax.legend()

# Animate
def update(i):
    tail = 200
    start = max(0, i - tail)
    line1.set_data(x1[start:i], y1[start:i])
    line1.set_3d_properties(z1[start:i])
    line2.set_data(x2[start:i], y2[start:i])
    line2.set_3d_properties(z2[start:i])
    return line1, line2

# ani = FuncAnimation(fig, update, frames=len(x1), interval=1, blit=False)
# ani.save("./fig/lorenz_diverging.mp4", writer="ffmpeg", fps=10)
plt.savefig(f'./fig/lorenz_divergence.png', dpi=300)
plt.show()


# Plot final snapshot after full simulation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

ax.plot(x1, y1, z1, color='blue', label=f'Initial: ({x1_0}, {y1_0}, {z1_0})')
ax.plot(x2, y2, z2, color='red', label=f'Perturbed: ({x2_0}, {y2_0}, {z2_0})')

ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)

ax.set_title("Divergence of Two Lorenz Trajectories")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.tight_layout()
plt.savefig(f'./fig/lorenz_divergence.png', dpi=300)
plt.close()

print("Saved snapshot as './fig/lorenz_divergence.png'")


# Compute distance between trajectories at each time step
x1_arr, y1_arr, z1_arr = np.array(x1), np.array(y1), np.array(z1)
x2_arr, y2_arr, z2_arr = np.array(x2), np.array(y2), np.array(z2)

distances = np.sqrt((x1_arr - x2_arr)**2 + (y1_arr - y2_arr)**2 + (z1_arr - z2_arr)**2)

# Avoid log(0)
distances[distances == 0] = 1e-16

# Take log of distances
log_dist = np.log(distances)

# Fit a line to the early-time behavior (before it saturates)
t = np.arange(0, end_time + dt, dt)
fit_end_index = 1999  # tweak this depending on when saturation begins

# Linear fit: log(d) ≈ lambda * t + c
from scipy.stats import linregress
slope, intercept, *_ = linregress(t[:fit_end_index], log_dist[:fit_end_index])

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(t[:fit_end_index], log_dist[:fit_end_index], label='log distance')
plt.plot(t[:fit_end_index], slope * t[:fit_end_index] + intercept, '--r', label=f'Linear fit (λ ≈ {slope:.3f})')
plt.xlabel('Time')
plt.ylabel('log(distance)')
plt.title('Estimation of Lyapunov Exponent from Divergence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./fig/lyapunov_estimate.png", dpi=300)
plt.close()

print(f"Largest Lyapunov exponent (approx): λ ≈ {slope:.3f}")
print("Saved plot as './fig/lyapunov_estimate.png'")
