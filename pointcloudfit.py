import numpy as np
from scipy.optimize import minimize
from tg43.I125 import I_125
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Dose_Rate1D(position: np.ndarray, xrange: np.ndarray, yrange: np.ndarray, zrange: np.ndarray, fuente) -> tuple:
    """
    Calculate the dose rate from a single particle at a given position.
    
    Parameters:
    - position: Coordinates of the particle.
    - xrange, yrange, zrange: Grid ranges for dose calculation in each dimension.
    - fuente: Particle source with dose properties.
    
    Returns:
    - point_cloud: 3D matrix of dose points.
    - dd: Dose distribution for the single particle.
    """
    X, Y, Z = np.meshgrid(xrange, yrange, zrange)
    Rs = np.sqrt((X - position[0])**2 + (Y - position[1])**2 + (Z - position[2])**2) / 10  # Convert to cm
    Sk = fuente.Sk

    geo_fun = np.where(Rs >= 1, 1 / Rs**2, (Rs**2 - 0.25 * fuente.length**2) / (1 - 0.25 * fuente.length**2))
    g_r = np.interp(Rs, fuente.RadialDoseFuntion['r(cm)'], fuente.RadialDoseFuntion['g(r)'])
    phi_r = np.interp(Rs, fuente.Phyani['r(cm)'], fuente.Phyani['phi(r)'])
    
    dd = Sk * fuente.DoseRateConstant * geo_fun * g_r * phi_r
    point_cloud = np.stack((X, Y, Z, dd), axis=3)
    
    return point_cloud, dd

def Dose_Distribution1D(seed_positions: np.ndarray, xrange: np.ndarray, yrange: np.ndarray, zrange: np.ndarray, fuente) -> np.ndarray:
    """
    Calculate the dose distribution for multiple particles.
    
    Parameters:
    - seed_positions: Array of particle positions.
    - xrange, yrange, zrange: Grid ranges for each dimension.
    - fuente: Particle source with dose properties.
    
    Returns:
    - Total dose distribution across the grid.
    """
    DoseRate = np.zeros((len(xrange), len(yrange), len(zrange)))
    for position in seed_positions:
        _, ddose = Dose_Rate1D(position, xrange, yrange, zrange, fuente)
        DoseRate += ddose
    return DoseRate

def gamma_value_loss(dose_target: np.ndarray, dose_current: np.ndarray, coordinates: np.ndarray, dose_criteria: float, dist_criteria: float) -> float:
    """
    Compute gamma loss for dose comparison.
    
    Parameters:
    - dose_target: Known dose field (target).
    - dose_current: Calculated dose field.
    - coordinates: Positions of the dose points.
    - dose_criteria, dist_criteria: Tolerance criteria for dose and spatial distance.
    
    Returns:
    - Total gamma loss for optimization.
    """
    gamma_values = []
    for dose_t, dose_c, coord in zip(dose_target, dose_current, coordinates):
        dose_diff = np.abs(dose_t - dose_c)
        dist_diff = np.linalg.norm(coord - coordinates, axis=1)
        gamma_candidate = np.sqrt((dose_diff / dose_criteria) ** 2 + (dist_diff / dist_criteria) ** 2)
        gamma_values.append(np.min(gamma_candidate))
    return np.mean(gamma_values)

def optimize_particle_positions(dose_target: np.ndarray, initial_positions: np.ndarray, xrange: np.ndarray, yrange: np.ndarray, zrange: np.ndarray, fuente, dose_criteria: float, dist_criteria: float) -> np.ndarray:
    """
    Optimize particle positions to minimize gamma loss.
    
    Parameters:
    - dose_target: Known dose distribution to match.
    - initial_positions: Initial particle positions.
    - xrange, yrange, zrange: Grid ranges for each dimension.
    - fuente: Source with dose properties.
    - dose_criteria, dist_criteria: Gamma criteria for dose and spatial distances.
    
    Returns:
    - Optimized particle positions.
    """
    def loss_function(positions):
        positions = positions.reshape(-1, 3)
        calculated_dose = Dose_Distribution1D(positions, xrange, yrange, zrange, fuente)
        return gamma_value_loss(dose_target.flatten(), calculated_dose.flatten(), positions, dose_criteria, dist_criteria)
    
    initial_params = initial_positions.flatten()
    result = minimize(loss_function, initial_params, method='L-BFGS-B')
    optimized_positions = result.x.reshape(-1, 3)
    
    return optimized_positions

# Example input setup
dose_target = np.load('beihangplandspc.npy')  # Known dose distribution #without position
initial_positions = np.random.rand(20, 3) * 10  # Initial positions for 24 particles
xrange = np.linspace(351.267591, 101.756, 64)
yrange = np.linspace(-124.756, -374.267591, 64)
zrange = np.linspace(-190, -1, 24)

# x_max = max(seedpos[:,0]) 
# x_min = min(seedpos[:,0]) 
# y_max = max(seedpos[:,1]) 
# y_min = min(seedpos[:,1]) 
# z_max = max(seedpos[:,2]) 
# z_min = min(seedpos[:,2]) 

# print(x_min,x_max,y_min,y_max,z_min,z_max) 

# Set criteria for gamma comparison
dose_criteria = 0.03
dist_criteria = 3.0

# Set seed type
I125seed = I_125(1,Sk=0.635)

# Optimize positions
optimized_positions = optimize_particle_positions(dose_target, initial_positions, xrange, yrange, zrange, I125seed, dose_criteria, dist_criteria)
print("Optimized particle positions:", optimized_positions)

def plot_point_cloud(points, title="Voxelized Point Cloud"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=2)
    ax.set_title(title)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()

plot_point_cloud(optimized_positions, "fitted Point Cloud")

np.savetxt('fit_opt_seed.txt',optimized_positions)