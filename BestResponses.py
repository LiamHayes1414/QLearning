import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

num_followers = 1
a = 0.1
b = 0.5
mc = 1

def price_equation(P, p):
    D = (num_followers * math.exp(-a * p)) + ((1 + b) * math.exp(-a * P)) + 1
    s_L = ((1 + b) * math.exp(-a * P)) / D

    return mc + (1 / (a * (1 - s_L))) - P


# Range of follower prices
follower_prices = np.linspace(12, 17, 100)
leader_prices = []

# Solve for the leader's best-response price at each follower price
for p in follower_prices:
    P = fsolve(price_equation, mc + 10, args=(p,))[0]
    leader_prices.append(P)

# Convert to NumPy arrays
x = np.array(follower_prices)
y = np.array(leader_prices)

# Identify where follower price > leader price
mask = x > y

# Plot
plt.figure(figsize=(8, 5))

# Blue where follower price <= leader price
plt.plot(
    x,
    np.where(~mask, y, np.nan),
    color="blue",
    label="Leader should charge higher"
)

# Red where follower price > leader price
plt.plot(
    x,
    np.where(mask, y, np.nan),
    color="red",
    label="Leader should undercut"
)
#put point about where results are
plt.scatter(14.6, 14.8, color="Green", s=40, zorder=5)
plt.annotate(
    "Approximate simulation result",
    xy=(14.6, 14.8),
    xytext=(13.5, 15),  # Position of the text box
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"),
    arrowprops=dict(arrowstyle="->", color="black")
)
plt.ylim(13, 16)

plt.xlabel("Follower price")
plt.ylabel("Leader best-response price")
plt.title(r"Leader's Best Response: $p_\ell = \frac{1}{\alpha(1-s_\ell)} + mc$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()