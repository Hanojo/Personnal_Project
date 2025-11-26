import numpy as np
import scipy.interpolate as interpolate
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from matplotlib import cm
import cmasher as cmr
from tqdm import tqdm
from typing import Tuple

# Constants
N_POINTS = 101               # Number of grid points in each direction (increased for better resolution)
DOMAIN_SIZE = 1.0            # Size of the square domain
N_TIME_STEPS = 250           # Number of time steps
TIME_STEP_LENGTH = 0.1      # Time step size (adjusted for stability)
KINEMATIC_VISCOSITY = 0.0001   # Kinematic viscosity (adjusted for bloom effect)
MAX_ITER_CG = 100            # Max iterations for conjugate gradient solver
PLOT_EVERY = 10              # Plot animation every N steps to optimize speed
PLOT_RESOLUTION = 2          # Downsampling factor for plots

# Grid setup
x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
X, Y = np.meshgrid(x, y)
coordinates = np.stack((X, Y), axis=-1)  # Shape: (N_POINTS, N_POINTS, 2)
dx = x[1] - x[0]
scalar_shape = (N_POINTS, N_POINTS)
vector_shape = (*scalar_shape, 2)
scalar_dof = np.prod(scalar_shape)
vector_dof = np.prod(vector_shape)

def forcing_function_vectorized(time: float, points: np.ndarray) -> np.ndarray:
    """Upward forcing in the lower center of the domain (bloom effect)."""
    x, y = points[..., 0], points[..., 1]
    forcing_x = np.zeros_like(x)
    forcing_y = (
        0.1
        * (1.0 + 0.1 * np.sin(2.0 * np.pi * time))  # Time-varying intensity
        * np.exp(-((x - 0.5) ** 2 + (y - 0.1) ** 2) / 0.01)  # Gaussian bloom centered at (0.5, 0.1)
    )
    return np.stack((forcing_x, forcing_y), axis=-1)

def laplace(field: np.ndarray) -> np.ndarray:
    """Compute Laplace operator (second derivative) with homogeneous Dirichlet BC."""
    laplacian = np.zeros_like(field)
    laplacian[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / dx**2 +
        (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / dx**2
    )
    return laplacian

def partial_derivative_x(field: np.ndarray) -> np.ndarray:
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * dx)
    return diff

def partial_derivative_y(field: np.ndarray) -> np.ndarray:
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2 * dx)
    return diff

def divergence(vector_field: np.ndarray) -> np.ndarray:
    div = partial_derivative_x(vector_field[..., 0]) + partial_derivative_y(vector_field[..., 1])
    # Enforce BC: divergence zero on boundaries
    div[[0, -1], :] = 0
    div[:, [0, -1]] = 0
    return div

def gradient(field: np.ndarray) -> np.ndarray:
    grad = np.stack((partial_derivative_x(field), partial_derivative_y(field)), axis=-1)
    # Enforce BC: gradient zero on boundaries for projection
    grad[[0, -1], :, :] = 0
    grad[:, [0, -1], :] = 0
    return grad

def curl_2d(vector_field: np.ndarray) -> np.ndarray:
    curl = partial_derivative_x(vector_field[..., 1]) - partial_derivative_y(vector_field[..., 0])
    curl[[0, -1], :] = 0
    curl[:, [0, -1]] = 0
    return curl

def compute_stream_function(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """Compute stream function by integrating velocity."""
    stream = np.zeros_like(u)
    stream[1:, :] = stream[:-1, :] - v[:-1, :] * dx
    stream[:, 1:] += u[:, :-1] * dx
    return stream

def advect(field: np.ndarray, vector_field: np.ndarray) -> np.ndarray:
    """Advect field using backtracing with RegularGridInterpolator."""
    backtraced_positions = np.clip(coordinates - TIME_STEP_LENGTH * vector_field, 0.0, DOMAIN_SIZE)
    
    # Swap to [y, x] order for interpolation (matches array indexing: rows=y, cols=x)
    points = backtraced_positions[..., [1, 0]]  # Shape: (N_POINTS, N_POINTS, 2) with [y_back, x_back]
    
    # Create interpolator: grid is (y, x), field is shaped (len(y), len(x), [components])
    interp = interpolate.RegularGridInterpolator(
        (y, x), field, method='linear', bounds_error=False, fill_value=0.0
    )
    
    # Interpolate: handles both scalar (returns (N,N)) and vector (returns (N,N,2))
    advected = interp(points)
    
    # Enforce homogeneous Dirichlet BC (u=v=0 on boundaries)
    if advected.ndim == 3:  # Vector field
        advected[[0, -1], :, :] = 0
        advected[:, [0, -1], :] = 0
    else:  # Scalar
        advected[[0, -1], :] = 0
        advected[:, [0, -1]] = 0
    
    return advected

def diffusion_operator(vector_field_flattened: np.ndarray) -> np.ndarray:
    vector_field = vector_field_flattened.reshape(vector_shape)
    diffusion = vector_field - KINEMATIC_VISCOSITY * TIME_STEP_LENGTH * laplace(vector_field)
    # Enforce BC
    diffusion[[0, -1], :, :] = 0
    diffusion[:, [0, -1], :] = 0
    return diffusion.flatten()

def poisson_operator(field_flattened: np.ndarray) -> np.ndarray:
    field = field_flattened.reshape(scalar_shape)
    poisson = laplace(field)
    # Enforce BC for pressure (Neumann-like, but simplified)
    poisson[[0, -1], :] = 0
    poisson[:, [0, -1]] = 0
    return poisson.flatten()

def main() -> None:
    velocities_prev = np.zeros(vector_shape)
    time_current = 0.0

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(5, 5), dpi=160)

    for i in tqdm(range(N_TIME_STEPS)):
        time_current += TIME_STEP_LENGTH

        forces = forcing_function_vectorized(time_current, coordinates)

        # (1) Apply Forces
        velocities_forces_applied = velocities_prev + TIME_STEP_LENGTH * forces

        # (2) Nonlinear convection (self-advection)
        velocities_advected = advect(velocities_forces_applied, velocities_forces_applied)

        # (3) Diffuse (implicit solve with CG)
        velocities_diffused = splinalg.cg(
            splinalg.LinearOperator((vector_dof, vector_dof), matvec=diffusion_operator),
            velocities_advected.flatten(),
            maxiter=MAX_ITER_CG,
            tol=1e-5
        )[0].reshape(vector_shape)

        # (4.1) Compute pressure correction (Poisson solve with CG)
        pressure = splinalg.cg(
            splinalg.LinearOperator((scalar_dof, scalar_dof), matvec=poisson_operator),
            divergence(velocities_diffused).flatten(),
            maxiter=MAX_ITER_CG,
            tol=1e-5
        )[0].reshape(scalar_shape)

        # (4.2) Project velocities to be incompressible
        velocities_projected = velocities_diffused - gradient(pressure)

        # Advance to next time step
        velocities_prev = velocities_projected

        # Real-time animation (plot every PLOT_EVERY steps)
        if i % PLOT_EVERY == 0:
            curl = curl_2d(velocities_projected)
            plt.clf()
            plt.contourf(X, Y, curl, cmap=cmr.redshift, levels=100)
            plt.quiver(X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                       Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                       velocities_projected[::PLOT_RESOLUTION, ::PLOT_RESOLUTION, 0],
                       velocities_projected[::PLOT_RESOLUTION, ::PLOT_RESOLUTION, 1],
                       color="dimgray")
            plt.title(f"Time: {time_current:.2f}")
            plt.draw()
            plt.pause(0.01)

    # Final comprehensive plots
    plt.close(fig)  # Close animation figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=160)

    # Plot 1: Pressure Field and Velocity Vectors
    ax = axs[0, 0]
    contour = ax.contourf(X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          pressure[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          levels=20, cmap="coolwarm")
    fig.colorbar(contour, ax=ax, label="Pressure")
    ax.quiver(X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
              Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
              velocities_projected[::PLOT_RESOLUTION, ::PLOT_RESOLUTION, 0],
              velocities_projected[::PLOT_RESOLUTION, ::PLOT_RESOLUTION, 1],
              color="black", scale=20, headwidth=3)
    ax.set_title("Pressure Field and Velocity Vectors")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Plot 2: Vorticity (Curl) Field
    ax = axs[0, 1]
    vorticity = curl_2d(velocities_projected)
    contour = ax.contourf(X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          vorticity[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          levels=20, cmap="RdBu")
    fig.colorbar(contour, ax=ax, label="Vorticity")
    ax.set_title("Vorticity Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Plot 3: Streamlines
    ax = axs[1, 0]
    stream_function = compute_stream_function(velocities_projected[..., 0], velocities_projected[..., 1], dx)
    ax.contour(X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
               Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
               stream_function[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
               levels=20, colors='black', linewidths=1)
    ax.set_title("Streamlines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Plot 4: Velocity Magnitude
    ax = axs[1, 1]
    velocity_magnitude = np.sqrt(velocities_projected[..., 0]**2 + velocities_projected[..., 1]**2)
    contour = ax.contourf(X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          velocity_magnitude[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
                          levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Velocity Magnitude")
    ax.set_title("Velocity Magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.suptitle("Stable Fluids Simulation Analysis (Final State)", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
