import numpy as np
import matplotlib.pyplot as plt
import xygame as xg

# def compute_angle_difference(angle1, angle2):
#     """Compute the angle difference considering periodic boundary conditions."""
#     diff = angle1 - angle2
#     return np.arctan2(np.sin(diff), np.cos(diff))
#     # return diff

# def detect_defects(spins):
#     """Detect vortices and antivortices in a 2D array of angles."""
#     vortices = []
#     Lx, Ly = spins.shape
    
#     for i in range(Lx - 1):
#         for j in range(Ly - 1):
#             # Angles at the corners of the plaquette
#             theta_00 = spins[i, j]
#             theta_10 = spins[i + 1, j]
#             theta_11 = spins[i + 1, j + 1]
#             theta_01 = spins[i, j + 1]

#             # Compute the angle differences around the plaquette
#             delta_theta1 = compute_angle_difference(theta_10, theta_00)
#             delta_theta2 = compute_angle_difference(theta_11, theta_10)
#             delta_theta3 = compute_angle_difference(theta_01, theta_11)
#             delta_theta4 = compute_angle_difference(theta_00, theta_01)

#             # Sum of the angle differences
#             circulation = delta_theta1 + delta_theta2 + delta_theta3 + delta_theta4

#             # Check if there's a vortex or antivortex
#             if circulation > np.pi:
#                 vortices.append((i + 0.5, j + 0.5))  # Vortex
#             elif circulation < -np.pi:
#                 vortices.append((i + 0.5, j + 0.5))  # Antivortex

#     return vortices

# Example: Create a 2D array of angles (spins) from 0 to 2*pi
Lx, Ly = 20, 20
x = np.linspace(-1, 1, Lx)
y = np.linspace(-1, 1, Ly)
X, Y = np.meshgrid(x, y)

angles = np.arctan2(Y, X)

# Create a 20x20 grid
# Lx, Ly = 20, 20
# x = np.linspace(-1, 1, Lx)
# y = np.linspace(-1, 1, Ly)
# X, Y = np.meshgrid(x, y)

# Offset the origin by 1 unit in x and 1 unit in y
# X_offset = X - 0.3
# Y_offset = Y - 0.3

# # Compute the angles for curl (rotational pattern)
# angles = np.arctan2(Y_offset, X_offset) + np.pi / 2


# Detect defects
defects = xg.detect_defects(angles)
print(defects)

# Plot the spin structure and the detected defects
X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly))
U = np.cos(angles)
V = np.sin(angles)

plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, pivot='mid')
for (x, y) in defects:
    plt.plot(x, y, 'ro' if x%2==0 else 'bo')  # red for vortices, blue for antivortices
plt.xlim(-1, Lx)
plt.ylim(-1, Ly)
plt.title('Spin Structure with Detected Defects')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()
