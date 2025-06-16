import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Logistic map function
def logistic_map(x, r):
    return r * x * (1 - x)

# Parameters
n_iter = 200  # iterations per r value
x0 = 0.5
r_values = np.linspace(0, 5, 400)  # smoother animation with more frames

# Set up the figure
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, n_iter)
ax.set_ylim(0, 1)
ax.set_xlabel("n (time)")
ax.set_ylabel("x_n")
title = ax.set_title("")

# Animation update function
def update(frame):
    r = r_values[frame]
    x = np.zeros(n_iter)
    x[0] = x0
    for n in range(1, n_iter):
        x[n] = logistic_map(x[n-1], r)
    line.set_data(np.arange(n_iter), x)
    title.set_text(f"Logistic Map: r = {r:.3f}")
    return line, title

ani = animation.FuncAnimation(fig, update, frames=len(r_values), blit=False, interval=40)
plt.show()
