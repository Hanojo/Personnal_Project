import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
N_POINTS_Y = 15
ASPECT_RATIO = 20
KINEMATIC_VISCOSITY = 0.01
TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 6000
PLOT_EVERY = 100

STEP_HEIGHT_POINTS = 7
STEP_WIDTH_POINTS = 60

N_PRESSURE_POISSON_ITERATIONS = 50


def check_stability(dt: float, dx: float, nu: float, safety: float = 0.5) -> None:
    """
    Simple diffusion stability check for explicit FTCS:
        nu * dt / dx^2 <= safety
    """
    diff_number = nu * dt / dx**2
    print(f"Stability check: ν dt / dx² = {diff_number:.4f}, safety limit = {safety:.4f}")
    if diff_number > safety:
        print("WARNING: Explicit diffusion step may be unstable. "
              "Reduce TIME_STEP_LENGTH or increase KINEMATIC_VISCOSITY.\n")


def map_to_vertex_centered(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map staggered u (Ny+1, Nx) and v (Ny, Nx+1) to vertex-centered (Ny, Nx).
    """
    u_vc = 0.5 * (u[1:, :] + u[:-1, :])
    v_vc = 0.5 * (v[:, 1:] + v[:, :-1])
    return u_vc, v_vc


def main():
    # -------------------------------------------------
    # Grid setup
    # -------------------------------------------------
    cell_length = 1.0 / (N_POINTS_Y - 1)
    n_points_x = (N_POINTS_Y - 1) * ASPECT_RATIO + 1

    x_range = np.linspace(0.0, 1.0 * ASPECT_RATIO, n_points_x)
    y_range = np.linspace(0.0, 1.0, N_POINTS_Y)
    coordinates_x, coordinates_y = np.meshgrid(x_range, y_range)

    # Stability info (for explicit diffusion)
    check_stability(TIME_STEP_LENGTH, cell_length, KINEMATIC_VISCOSITY)

    # Precompute step mask on vertex-centered grid (for plotting)
    step_mask = np.zeros((N_POINTS_Y, n_points_x), dtype=bool)
    step_mask[: STEP_HEIGHT_POINTS + 1, : STEP_WIDTH_POINTS + 1] = True

    # -------------------------------------------------
    # Initial condition (staggered)
    # -------------------------------------------------
    velocity_x_prev = np.ones((N_POINTS_Y + 1, n_points_x))
    velocity_x_prev[: (STEP_HEIGHT_POINTS + 1), :] = 0.0

    # Top Edge
    velocity_x_prev[-1, :] = -velocity_x_prev[-2, :]

    # Top Edge of the step
    velocity_x_prev[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] = \
        -velocity_x_prev[STEP_HEIGHT_POINTS + 1, 1:STEP_WIDTH_POINTS]

    # Right Edge of the step
    velocity_x_prev[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

    # Bottom Edge of the domain
    velocity_x_prev[0, (STEP_WIDTH_POINTS + 1):-1] = \
        -velocity_x_prev[1, (STEP_WIDTH_POINTS + 1):-1]

    # Values inside of the step
    velocity_x_prev[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

    velocity_y_prev = np.zeros((N_POINTS_Y, n_points_x + 1))
    pressure_prev = np.zeros((N_POINTS_Y + 1, n_points_x + 1))

    # Pre-allocate arrays
    velocity_x_tent = np.zeros_like(velocity_x_prev)
    velocity_x_next = np.zeros_like(velocity_x_prev)
    velocity_y_tent = np.zeros_like(velocity_y_prev)
    velocity_y_next = np.zeros_like(velocity_y_prev)

    # Diagnostics
    u_residual_history = []
    v_residual_history = []
    divergence_L2_history = []

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6))

    # -------------------------------------------------
    # Time-stepping loop
    # -------------------------------------------------
    for iter in tqdm(range(N_TIME_STEPS)):
        # -----------------------------
        # 1) Tentative u-velocity
        # -----------------------------
        diffusion_x = KINEMATIC_VISCOSITY * (
            (
                velocity_x_prev[1:-1, 2:] +
                velocity_x_prev[2:, 1:-1] +
                velocity_x_prev[1:-1, :-2] +
                velocity_x_prev[:-2, 1:-1] -
                4 * velocity_x_prev[1:-1, 1:-1]
            ) / cell_length**2
        )

        convection_x = (
            (
                velocity_x_prev[1:-1, 2:]**2 -
                velocity_x_prev[1:-1, :-2]**2
            ) / (2 * cell_length)
            +
            (
                velocity_y_prev[1:, 1:-2] +
                velocity_y_prev[1:, 2:-1] +
                velocity_y_prev[:-1, 1:-2] +
                velocity_y_prev[:-1, 2:-1]
            ) / 4.0
            *
            (
                velocity_x_prev[2:, 1:-1] -
                velocity_x_prev[:-2, 1:-1]
            ) / (2 * cell_length)
        )

        pressure_gradient_x = (
            pressure_prev[1:-1, 2:-1] -
            pressure_prev[1:-1, 1:-2]
        ) / cell_length

        velocity_x_tent[1:-1, 1:-1] = (
            velocity_x_prev[1:-1, 1:-1] +
            TIME_STEP_LENGTH * (
                -pressure_gradient_x +
                diffusion_x -
                convection_x
            )
        )

        # u-BCs (tentative)
        # Inflow
        velocity_x_tent[STEP_HEIGHT_POINTS + 1:-1, 0] = 1.0

        # Outflow (mass-flux correction)
        inflow_mass_rate_tent = np.sum(velocity_x_tent[STEP_HEIGHT_POINTS + 1:-1, 0])
        outflow_mass_rate_tent = np.sum(velocity_x_tent[1:-1, -2])
        velocity_x_tent[1:-1, -1] = (
            velocity_x_tent[1:-1, -2] *
            inflow_mass_rate_tent / outflow_mass_rate_tent
        )

        # Top edge of the step
        velocity_x_tent[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] = \
            -velocity_x_tent[STEP_HEIGHT_POINTS + 1, 1:STEP_WIDTH_POINTS]

        # Right edge of the step
        velocity_x_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

        # Bottom edge of the domain
        velocity_x_tent[0, (STEP_WIDTH_POINTS + 1):-1] = \
            -velocity_x_tent[1, (STEP_WIDTH_POINTS + 1):-1]

        # Top edge of the domain
        velocity_x_tent[-1, :] = -velocity_x_tent[-2, :]

        # Inside the step
        velocity_x_tent[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        # -----------------------------
        # 2) Tentative v-velocity
        # -----------------------------
        diffusion_y = KINEMATIC_VISCOSITY * (
            (
                velocity_y_prev[1:-1, 2:] +
                velocity_y_prev[2:, 1:-1] +
                velocity_y_prev[1:-1, :-2] +
                velocity_y_prev[:-2, 1:-1] -
                4 * velocity_y_prev[1:-1, 1:-1]
            ) / cell_length**2
        )

        convection_y = (
            (
                velocity_x_prev[2:-1, 1:] +
                velocity_x_prev[2:-1, :-1] +
                velocity_x_prev[1:-2, 1:] +
                velocity_x_prev[1:-2, :-1]
            ) / 4.0
            *
            (
                velocity_y_prev[1:-1, 2:] -
                velocity_y_prev[1:-1, :-2]
            ) / (2 * cell_length)
            +
            (
                velocity_y_prev[2:, 1:-1]**2 -
                velocity_y_prev[:-2, 1:-1]**2
            ) / (2 * cell_length)
        )

        pressure_gradient_y = (
            pressure_prev[2:-1, 1:-1] -
            pressure_prev[1:-2, 1:-1]
        ) / cell_length

        velocity_y_tent[1:-1, 1:-1] = (
            velocity_y_prev[1:-1, 1:-1] +
            TIME_STEP_LENGTH * (
                -pressure_gradient_y +
                diffusion_y -
                convection_y
            )
        )

        # v-BCs (tentative)
        # Inflow
        velocity_y_tent[STEP_HEIGHT_POINTS + 1:-1, 0] = \
            -velocity_y_tent[STEP_HEIGHT_POINTS + 1:-1, 1]

        # Outflow
        velocity_y_tent[1:-1, -1] = velocity_y_tent[1:-1, -2]

        # Top edge of the step
        velocity_y_tent[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS + 1] = 0.0

        # Right edge of the step
        velocity_y_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = \
            -velocity_y_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS + 1]

        # Bottom edge of the domain
        velocity_y_tent[0, STEP_WIDTH_POINTS + 1:] = 0.0

        # Top edge of the domain
        velocity_y_tent[-1, :] = 0.0

        # Inside the step
        velocity_y_tent[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        # -----------------------------
        # 3) Divergence (for Poisson RHS)
        # -----------------------------
        divergence = (
            (
                velocity_x_tent[1:-1, 1:] -
                velocity_x_tent[1:-1, :-1]
            ) / cell_length
            +
            (
                velocity_y_tent[1:, 1:-1] -
                velocity_y_tent[:-1, 1:-1]
            ) / cell_length
        )
        pressure_poisson_rhs = divergence / TIME_STEP_LENGTH

        # Track divergence norm
        div_L2 = np.linalg.norm(divergence) / np.sqrt(divergence.size)
        divergence_L2_history.append(div_L2)

        # -----------------------------
        # 4) Pressure correction Poisson
        # -----------------------------
        pressure_correction_prev = np.zeros_like(pressure_prev)
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            pressure_correction_next = np.zeros_like(pressure_correction_prev)
            pressure_correction_next[1:-1, 1:-1] = 0.25 * (
                pressure_correction_prev[1:-1, 2:] +
                pressure_correction_prev[2:, 1:-1] +
                pressure_correction_prev[1:-1, :-2] +
                pressure_correction_prev[:-2, 1:-1] -
                cell_length**2 * pressure_poisson_rhs
            )

            # Pressure BCs (homogeneous Neumann except outlet)
            # Inflow
            pressure_correction_next[STEP_HEIGHT_POINTS + 1:-1, 0] = \
                pressure_correction_next[STEP_HEIGHT_POINTS + 1:-1, 1]

            # Outflow (homogeneous Dirichlet)
            pressure_correction_next[1:-1, -1] = \
                -pressure_correction_next[1:-1, -2]

            # Top edge of the step
            pressure_correction_next[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS + 1] = \
                pressure_correction_next[STEP_HEIGHT_POINTS + 1, 1:STEP_WIDTH_POINTS + 1]

            # Right edge of the step
            pressure_correction_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = \
                pressure_correction_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS + 1]

            # Bottom edge of the domain
            pressure_correction_next[0, STEP_WIDTH_POINTS + 1:-1] = \
                pressure_correction_next[1, STEP_WIDTH_POINTS + 1:-1]

            # Top edge of the domain
            pressure_correction_next[-1, :] = pressure_correction_next[-2, :]

            # Inside the step
            pressure_correction_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

            pressure_correction_prev = pressure_correction_next

        # -----------------------------
        # 5) Pressure update
        # -----------------------------
        pressure_next = pressure_prev + pressure_correction_next

        # -----------------------------
        # 6) Velocity correction
        # -----------------------------
        pressure_correction_gradient_x = (
            pressure_correction_next[1:-1, 2:-1] -
            pressure_correction_next[1:-1, 1:-2]
        ) / cell_length

        velocity_x_next[1:-1, 1:-1] = (
            velocity_x_tent[1:-1, 1:-1] -
            TIME_STEP_LENGTH * pressure_correction_gradient_x
        )

        pressure_correction_gradient_y = (
            pressure_correction_next[2:-1, 1:-1] -
            pressure_correction_next[1:-2, 1:-1]
        ) / cell_length

        velocity_y_next[1:-1, 1:-1] = (
            velocity_y_tent[1:-1, 1:-1] -
            TIME_STEP_LENGTH * pressure_correction_gradient_y
        )

        # Re-apply BCs to corrected velocities
        # u:
        velocity_x_next[STEP_HEIGHT_POINTS + 1:-1, 0] = 1.0

        inflow_mass_rate_next = np.sum(velocity_x_next[STEP_HEIGHT_POINTS + 1:-1, 0])
        outflow_mass_rate_next = np.sum(velocity_x_next[1:-1, -2])
        velocity_x_next[1:-1, -1] = (
            velocity_x_next[1:-1, -2] *
            inflow_mass_rate_next / outflow_mass_rate_next
        )

        velocity_x_next[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] = \
            -velocity_x_next[STEP_HEIGHT_POINTS + 1, 1:STEP_WIDTH_POINTS]

        velocity_x_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

        velocity_x_next[0, STEP_WIDTH_POINTS + 1:-1] = \
            -velocity_x_next[1, STEP_WIDTH_POINTS + 1:-1]

        velocity_x_next[-1, :] = -velocity_x_next[-2, :]

        velocity_x_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        # v:
        velocity_y_next[STEP_HEIGHT_POINTS + 1:-1, 0] = \
            -velocity_y_next[STEP_HEIGHT_POINTS + 1:-1, 1]

        velocity_y_next[1:-1, -1] = velocity_y_next[1:-1, -2]

        velocity_y_next[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS + 1] = 0.0

        velocity_y_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = \
            -velocity_y_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS + 1]

        velocity_y_next[0, STEP_WIDTH_POINTS + 1:] = 0.0

        velocity_y_next[-1, :] = 0.0

        velocity_y_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        # -----------------------------
        # Diagnostics: residuals
        # -----------------------------
        u_res = np.linalg.norm(velocity_x_next - velocity_x_prev) / np.sqrt(velocity_x_next.size)
        v_res = np.linalg.norm(velocity_y_next - velocity_y_prev) / np.sqrt(velocity_y_next.size)
        u_residual_history.append(u_res)
        v_residual_history.append(v_res)

        # -----------------------------
        # Advance in time
        # -----------------------------
        velocity_x_prev = velocity_x_next.copy()
        velocity_y_prev = velocity_y_next.copy()
        pressure_prev = pressure_next.copy()

        # -----------------------------
        # Live visualization
        # -----------------------------
        if iter % PLOT_EVERY == 0:
            u_vc, v_vc = map_to_vertex_centered(velocity_x_prev, velocity_y_prev)

            # Zero velocities inside the step for visualization
            u_vc[step_mask] = 0.0
            v_vc[step_mask] = 0.0

            plt.contourf(
                coordinates_x,
                coordinates_y,
                u_vc,
                levels=10,
                cmap=cmr.amber,
                vmin=-1.5,
                vmax=1.5,
            )
            plt.colorbar(label="u-velocity")

            plt.quiver(
                coordinates_x[:, ::6],
                coordinates_y[:, ::6],
                u_vc[:, ::6],
                v_vc[:, ::6],
                alpha=0.4,
            )

            plt.title(f"Backward-facing step flow (t = {iter * TIME_STEP_LENGTH:.3f})")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout()
            plt.pause(0.1)
            plt.clf()

    # -------------------------------------------------
    # Final analysis plots
    # -------------------------------------------------
    u_vc, v_vc = map_to_vertex_centered(velocity_x_prev, velocity_y_prev)
    u_vc[step_mask] = 0.0
    v_vc[step_mask] = 0.0

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=120)
    fig.suptitle("Backward-Facing Step Flow – Final State and Convergence", fontsize=14)

    # (1) Final u-velocity field + quiver
    ax = axs[0, 0]
    c = ax.contourf(
        coordinates_x,
        coordinates_y,
        u_vc,
        levels=20,
        cmap=cmr.amber,
        vmin=-1.5,
        vmax=1.5,
    )
    fig.colorbar(c, ax=ax, label="u-velocity")
    ax.quiver(
        coordinates_x[:, ::6],
        coordinates_y[:, ::6],
        u_vc[:, ::6],
        v_vc[:, ::6],
        alpha=0.5,
        color="k",
        scale=50,
    )
    ax.set_title("Final u-velocity field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (2) Streamlines
    ax = axs[0, 1]
    ax.streamplot(
        coordinates_x,
        coordinates_y,
        u_vc,
        v_vc,
        density=1.5,
        color=u_vc,
        cmap=cmr.iceburn,
    )
    ax.set_title("Streamlines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (3) Residuals and divergence
    ax = axs[1, 0]
    steps = np.arange(N_TIME_STEPS)
    ax.semilogy(steps, u_residual_history, label="u-residual", color="deepskyblue")
    ax.semilogy(steps, v_residual_history, label="v-residual", color="lime")
    ax.semilogy(steps, divergence_L2_history, label="||div u||", color="orange")
    ax.set_title("Convergence history")
    ax.set_xlabel("time step")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # (4) u-profiles at several x-locations
    ax = axs[1, 1]
    candidate_indices = [5, 40, 80, 180]
    profile_indices = [i for i in candidate_indices if i < n_points_x - 1]
    for j in profile_indices:
        ax.plot(
            u_vc[:, j],
            y_range,
            label=f"x = {x_range[j]:.1f}",
        )
    ax.set_title("u-velocity profiles at selected x")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
