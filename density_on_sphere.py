import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import gamma
from scipy.stats import t

# --- Your Provided Functions ---
def log_density(x, alpha=1): #(Product exponential-power)
    scale = np.sqrt(gamma(1/alpha) / gamma(3/alpha))
    return -np.sum((np.abs(x) / scale) ** alpha, axis=-1)

def stereographic_projection(point, R):
    'SP: z to x'
    point = np.asarray(point)
    # Extract the last coordinate (z_{d+1})
    p_last = point[-1]
    # Compute scaling factor (1 - z_{d+1})
    scale = 1.0 - p_last
    # Project onto equatorial plane
    projected_points = R * point[:-1] / scale
    return projected_points

def inverse_stereographic_projection(point, R):
    'SP^{-1}: x to z'
    point = np.asarray(point)
    # Compute squared norms of the points
    q_squared_norms = np.linalg.norm(point)**2
    # Compute scaling factor
    denominator = R ** 2 + q_squared_norms
    # Project back to the sphere
    p_components = (2 * R * point) / denominator
    p_last = (q_squared_norms - R ** 2) / denominator
    # Combine into (d+1)-dimensional points
    projected_point = np.append(p_components, p_last)
    return projected_point


# --- Plotting the Heatmap on the Sphere ---
R_val = np.sqrt(2)

# 1. Create a spherical meshgrid
# We start latitude slightly above 0 to avoid hitting the exact North Pole (z=1),
# which prevents a division-by-zero error in the projection factor (1 - z).
phi = np.linspace(0, 2 * np.pi, 200)   # Longitude
theta = np.linspace(1e-4, np.pi, 100)  # Latitude
Phi, Theta = np.meshgrid(phi, theta)

X = np.sin(Theta) * np.cos(Phi)
Y = np.sin(Theta) * np.sin(Phi)
Z = np.cos(Theta)

# Pack into an array of shape (3, 100, 200)
sphere_points = np.array([X, Y, Z])

# 2. Project 3D sphere points onto the 2D plane using your function
plane_points = stereographic_projection(sphere_points, R=R_val)

# 3. Evaluate the 2D Density
# Your log_density expects the spatial coordinates on the last axis.
# We move the (x,y) axis from index 0 to the end, resulting in shape (100, 200, 2)
plane_points_reshaped = np.moveaxis(plane_points, 0, -1)
plane_log_density = log_density(plane_points_reshaped, alpha=1)
plane_density = np.exp(plane_log_density)

# 4. Apply Jacobian for true spherical density
# Area correction: dA_plane / dA_sphere = R^2 / (1 - z)^2
jacobian = (R_val**2) / ((1.0 - Z)**2)
spherical_density = plane_density * jacobian

# Normalize density between 0 and 1 strictly for color mapping purposes
norm_density = (spherical_density - spherical_density.min()) / (spherical_density.max() - spherical_density.min())

# 5. Render the 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Map the normalized density values to a colormap (Magma offers great contrast)
colors = cm.magma(norm_density)

surf = ax.plot_surface(X, Y, Z, facecolors=colors, shade=False, antialiased=True)

# Add a color bar
mappable = cm.ScalarMappable(cmap=cm.magma)
mappable.set_array(spherical_density)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, label='True Spherical Probability Density')

# Formatting
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()
plt.title(f"Stereographic Density Heatmap ($R=\\sqrt{{2}}$)")
plt.show()