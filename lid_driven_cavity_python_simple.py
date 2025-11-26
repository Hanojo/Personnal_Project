import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Tuple

# Constants
N_POINTS = 41                # Number of grid points in each direction
DOMAIN_SIZE = 1.0            # Size of the square domain
N_ITERATIONS = 500           # Number of time steps
TIME_STEP_LENGTH = 0.001     # Time step size
KINEMATIC_VISCOSITY = 0.1    # Kinematic viscosity of the fluid
DENSITY = 1.0                # Fluid density
HORIZONTAL_VELOCITY_TOP = 1.0  # Velocity of the moving lid
N_PRESSURE_POISSON_ITERATIONS = 50  # Number of iterations for pressure solver
STABILITY_SAFETY_FACTOR = 0.5  # Safety factor for stability condition
PLOT_RESOLUTION = 2           # Downsampling factor for plots

def initialize_grid(n_points: int, domain_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the computational grid and solution arrays.

    Args:
        n_points: Number of grid points in each direction
        domain_size: Size of the square domain

    Returns:
        Tuple containing X, Y meshgrid coordinates, and initialized u, v, p arrays
    """
    element_length = domain_size / (n_points - 1)
    x = np.linspace(0.0, domain_size, n_points)
    y = np.linspace(0.0, domain_size, n_points)
    X, Y = np.meshgrid(x, y)

    # Initialize velocity and pressure fields
    u_prev = np.zeros_like(X)  # x-velocity
    v_prev = np.zeros_like(X)  # y-velocity
    p_prev = np.zeros_like(X)  # pressure

    return X, Y, u_prev, v_prev, p_prev, element_length

def central_difference_x(f: np.ndarray, dx: float) -> np.ndarray:
    """Compute central difference in x-direction.

    Args:
        f: Field to differentiate
        dx: Grid spacing

    Returns:
        Central difference approximation of ∂f/∂x
    """
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * dx)
    return diff

def central_difference_y(f: np.ndarray, dx: float) -> np.ndarray:
    """Compute central difference in y-direction.

    Args:
        f: Field to differentiate
        dx: Grid spacing

    Returns:
        Central difference approximation of ∂f/∂y
    """
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * dx)
    return diff

def laplace(f: np.ndarray, dx: float) -> np.ndarray:
    """Compute the Laplacian of a 2D field.

    Args:
        f: Field to compute Laplacian of
        dx: Grid spacing

    Returns:
        Discrete Laplacian of f
    """
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[1:-1, 0:-2] + f[0:-2, 1:-1] - 4 * f[1:-1, 1:-1] +
        f[1:-1, 2:] + f[2:, 1:-1]
    ) / (dx**2)
    return diff

def apply_velocity_boundary_conditions(u: np.ndarray, v: np.ndarray, u_top: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply velocity boundary conditions.

    Homogeneous Dirichlet BC everywhere except for horizontal velocity at top.

    Args:
        u: x-velocity field
        v: y-velocity field
        u_top: Velocity at the top boundary

    Returns:
        Tuple of velocity fields with boundary conditions applied
    """
    # Homogeneous Dirichlet BC everywhere except for horizontal velocity at top
    u[0, :] = 0.0    # Bottom boundary
    u[:, 0] = 0.0    # Left boundary
    u[:, -1] = 0.0   # Right boundary
    u[-1, :] = u_top # Top boundary (moving lid)

    v[0, :] = 0.0    # Bottom boundary
    v[:, 0] = 0.0    # Left boundary
    v[:, -1] = 0.0   # Right boundary
    v[-1, :] = 0.0   # Top boundary

    return u, v

def apply_pressure_boundary_conditions(p: np.ndarray) -> np.ndarray:
    """Apply pressure boundary conditions.

    Homogeneous Neumann BC everywhere except for top (Dirichlet).

    Args:
        p: Pressure field

    Returns:
        Pressure field with boundary conditions applied
    """
    # Homogeneous Neumann BC everywhere except for top (Dirichlet)
    p[:, -1] = p[:, -2]  # Right boundary
    p[0, :] = p[1, :]    # Bottom boundary
    p[:, 0] = p[:, 1]    # Left boundary
    p[-1, :] = 0.0       # Top boundary (Dirichlet)
    return p

def solve_pressure_poisson(p_prev: np.ndarray, rhs: np.ndarray, dx: float, n_iter: int) -> np.ndarray:
    """Solve the pressure Poisson equation using Jacobi iteration.

    Args:
        p_prev: Pressure field from previous iteration
        rhs: Right-hand side of the Poisson equation
        dx: Grid spacing
        n_iter: Number of iterations to perform

    Returns:
        Updated pressure field
    """
    p_next = np.zeros_like(p_prev)

    for _ in range(n_iter):
        # Jacobi iteration for pressure Poisson equation
        p_next[1:-1, 1:-1] = 0.25 * (
            p_prev[1:-1, 0:-2] + p_prev[0:-2, 1:-1] +
            p_prev[1:-1, 2:] + p_prev[2:, 1:-1] -
            dx**2 * rhs[1:-1, 1:-1]
        )

        # Apply pressure boundary conditions
        p_next = apply_pressure_boundary_conditions(p_next)
        p_prev = p_next

    return p_next

def check_stability(dt: float, dx: float, nu: float, safety_factor: float) -> None:
    """Check if the time step satisfies the stability condition.

    Args:
        dt: Time step
        dx: Grid spacing
        nu: Kinematic viscosity
        safety_factor: Safety factor for stability

    Raises:
        RuntimeError: If time step exceeds stability limit
    """
    max_dt = 0.5 * dx**2 / nu
    if dt > safety_factor * max_dt:
        raise RuntimeError(f"Time step {dt:.2e} exceeds stability limit {safety_factor * max_dt:.2e}")

def compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """Compute the vorticity field from velocity components.

    Args:
        u: x-velocity field
        v: y-velocity field
        dx: Grid spacing

    Returns:
        Vorticity field (∂v/∂x - ∂u/∂y)
    """
    dv_dx = central_difference_x(v, dx)
    du_dy = central_difference_y(u, dx)
    return dv_dx - du_dy

def compute_stream_function(u: np.ndarray, v: np.ndarray, dx: float, n_iter: int = 1000) -> np.ndarray:
    """Compute the stream function from velocity components.

    Args:
        u: x-velocity field
        v: y-velocity field
        dx: Grid spacing
        n_iter: Number of iterations for Poisson solver

    Returns:
        Stream function field
    """
    # Initialize stream function
    psi = np.zeros_like(u)

    # Compute right-hand side (vorticity)
    vorticity = compute_vorticity(u, v, dx)
    rhs = -vorticity

    # Solve Poisson equation for stream function
    for _ in range(n_iter):
        psi[1:-1, 1:-1] = 0.25 * (
            psi[1:-1, 0:-2] + psi[0:-2, 1:-1] +
            psi[1:-1, 2:] + psi[2:, 1:-1] -
            dx**2 * rhs[1:-1, 1:-1]
        )

        # Apply boundary conditions (stream function is constant on boundaries)
        psi[0, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0
        psi[-1, :] = 0.0

    return psi

def main():
    # Initialize grid and solution arrays
    X, Y, u_prev, v_prev, p_prev, dx = initialize_grid(N_POINTS, DOMAIN_SIZE)

    # Check stability condition
    check_stability(TIME_STEP_LENGTH, dx, KINEMATIC_VISCOSITY, STABILITY_SAFETY_FACTOR)

    # Main time-stepping loop
    for _ in tqdm(range(N_ITERATIONS)):
        # Compute spatial derivatives
        du_dx = central_difference_x(u_prev, dx)
        du_dy = central_difference_y(u_prev, dx)
        dv_dx = central_difference_x(v_prev, dx)
        dv_dy = central_difference_y(v_prev, dx)

        # Compute Laplacians
        laplace_u = laplace(u_prev, dx)
        laplace_v = laplace(v_prev, dx)

        # Tentative velocity step (without pressure)
        u_tent = u_prev + TIME_STEP_LENGTH * (
            - (u_prev * du_dx + v_prev * du_dy) +
            KINEMATIC_VISCOSITY * laplace_u
        )
        v_tent = v_prev + TIME_STEP_LENGTH * (
            - (u_prev * dv_dx + v_prev * dv_dy) +
            KINEMATIC_VISCOSITY * laplace_v
        )

        # Apply velocity boundary conditions
        u_tent, v_tent = apply_velocity_boundary_conditions(u_tent, v_tent, HORIZONTAL_VELOCITY_TOP)

        # Compute divergence for pressure Poisson equation
        div_u = central_difference_x(u_tent, dx) + central_difference_y(v_tent, dx)
        rhs = (DENSITY / TIME_STEP_LENGTH) * div_u

        # Solve pressure Poisson equation
        p_next = solve_pressure_poisson(p_prev, rhs, dx, N_PRESSURE_POISSON_ITERATIONS)

        # Compute pressure gradients
        dp_dx = central_difference_x(p_next, dx)
        dp_dy = central_difference_y(p_next, dx)

        # Correct velocities with pressure gradient
        u_next = u_tent - (TIME_STEP_LENGTH / DENSITY) * dp_dx
        v_next = v_tent - (TIME_STEP_LENGTH / DENSITY) * dp_dy

        # Apply velocity boundary conditions again
        u_next, v_next = apply_velocity_boundary_conditions(u_next, v_next, HORIZONTAL_VELOCITY_TOP)

        # Update solution for next time step
        u_prev, v_prev, p_prev = u_next, v_next, p_next

    # Create figure with multiple subplots
    plt.figure(figsize=(15, 10))
    # Removed plt.style.use("dark_background") to use default white background for better visibility

    # Plot 1: Pressure field and velocity vectors
    plt.subplot(2, 2, 1)
    contour = plt.contourf(
        X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        p_next[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        levels=20,
        cmap="coolwarm"
    )
    plt.colorbar(contour, label="Pressure")
    plt.quiver(
        X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        u_next[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        v_next[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        color="black",  # Kept black as requested
        scale=20,
        headwidth=3
    )
    plt.title("Pressure Field and Velocity Vectors")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot 2: Vorticity field
    plt.subplot(2, 2, 2)
    vorticity = compute_vorticity(u_next, v_next, dx)
    contour = plt.contourf(
        X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        vorticity[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        levels=20,
        cmap="RdBu"
    )
    plt.colorbar(contour, label="Vorticity")
    plt.title("Vorticity Field")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot 3: Streamlines
    plt.subplot(2, 2, 3)
    stream_function = compute_stream_function(u_next, v_next, dx)
    plt.contour(
        X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        stream_function[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        levels=20,
        colors='black',  # Changed to black for visibility on white background
        linewidths=1
    )
    plt.title("Streamlines")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot 4: Velocity magnitude
    plt.subplot(2, 2, 4)
    velocity_magnitude = np.sqrt(u_next**2 + v_next**2)
    contour = plt.contourf(
        X[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        Y[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        velocity_magnitude[::PLOT_RESOLUTION, ::PLOT_RESOLUTION],
        levels=20,
        cmap="viridis"
    )
    plt.colorbar(contour, label="Velocity Magnitude")
    plt.title("Velocity Magnitude")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.suptitle("Lid-Driven Cavity Flow Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
