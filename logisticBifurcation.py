import numpy as np
import matplotlib.pyplot as plt

# Logistic map function
def logistic_map(x, r):
    return r * x * (1 - x)

# Parameters
r_values = np.linspace(0, 4, 10000)   # range of r (0 to 4), with high resolution
n_iter = 1000                         # total iterations per r
n_transient = 200                     # number of transient steps to ignore
x0 = 0.5                              # initial condition

# Set up plot data
r_plot = []
x_plot = []
lyapunov = []

for r in r_values:
    x = x0
    le_sum = 0
    # Run the map for some transient iterations to reach steady-state
    for _ in range(n_transient):
        x = logistic_map(x, r)
    # Now collect the remaining iterations
    for _ in range(n_iter - n_transient):
        x = logistic_map(x, r)
        r_plot.append(r)
        x_plot.append(x)
        le_sum += np.log(abs(r * (1 - 2 * x)))
    lyapunov.append(le_sum/(n_iter - n_transient))

# # Plotting
# plt.figure(figsize=(10, 7))
# plt.plot(r_plot, x_plot, ',k', alpha=0.25)
# plt.title("Bifurcation Diagram of the Logistic Map")
# plt.xlabel("r")
# plt.ylabel("x (long-term behavior)")
# plt.grid(False)
# plt.tight_layout()
# # plt.savefig("./fig/LogisticBifurcation.jpg", dpi=300)
# plt.show()


# # Plotting Lyapunov exponent
# plt.figure(figsize=(10, 5))
# plt.plot(r_values, lyapunov, 'b-', linewidth=1)
# plt.axhline(0, color='k', lw=0.5, linestyle='--')
# plt.title("Lyapunov Exponent of the Logistic Map")
# plt.xlabel("r")
# plt.ylabel("Lyapunov Exponent λ")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./fig/logDiffLyapunov.jpg", dpi=200)
# plt.show()


# Create a figure with two subplots (stacked vertically)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# --- Bifurcation Diagram ---
ax1.plot(r_plot, x_plot, ',k', alpha=0.25)
ax1.set_title("Bifurcation Diagram of the Logistic Map")
ax1.set_ylabel("x (long-term behavior)")
ax1.grid(False)

# --- Lyapunov Exponent Plot ---
ax2.plot(r_values, lyapunov, 'b-', linewidth=1)
ax2.axhline(0, color='k', lw=0.5, linestyle='--')
ax2.set_title("Lyapunov Exponent of the Logistic Map")
ax2.set_xlabel("r")
ax2.set_ylabel("Lyapunov Exponent λ")
ax2.grid(True)

# Tidy up layout and save (optional)
plt.tight_layout()
plt.savefig("./fig/logBifurLyap.png", dpi=300)
plt.show()