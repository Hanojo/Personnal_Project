import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
N_POINTS = 51
KINEMATIC_VISCOSITY = 0.01
TIME_STEP_LENGTH = 0.02
N_TIME_STEPS = 1500

PRESSURE_GRADIENT = np.array([-1.0, 0.0])  # constant d p / d x (negative = drives flow in +x)

PLOT_EVERY = 5      # update live plot every N steps (for speed)
QUIVER_SKIP = 1     # quiver downsampling (1 = show all, 2 = every 2nd, etc.)


def main():
    element_length = 1.0 / (N_POINTS - 1)

    x_range = np.linspace(0.0, 1.0, N_POINTS)
    y_range = np.linspace(0.0, 1.0, N_POINTS)
    coordinates_x, coordinates_y = np.meshgrid(x_range, y_range)

    def central_difference_x_periodic(field: np.ndarray) -> np.ndarray:
        """Central difference in x with periodic boundaries."""
        return (
            np.roll(field, shift=1, axis=1)
            -
            np.roll(field, shift=-1, axis=1)
        ) / (2 * element_length)

    def laplace_periodic(field: np.ndarray) -> np.ndarray:
        """
        5-point stencil Laplacian with periodic BCs in x and y.
        Note: y is physically no-slip only via setting velocity to zero at walls.
        """
        return (
            np.roll(field, shift=1, axis=1)
            +
            np.roll(field, shift=1, axis=0)
            +
            np.roll(field, shift=-1, axis=1)
            +
            np.roll(field, shift=-1, axis=0)
            -
            4.0 * field
        ) / (element_length ** 2)

    # -----------------------------
    # Initial condition
    # -----------------------------
    velocity_x_prev = np.ones((N_POINTS, N_POINTS))
    velocity_x_prev[0, :] = 0.0     # bottom wall
    velocity_x_prev[-1, :] = 0.0    # top wall

    # Diagnostics storage
    residual_history = []
    centerline_history = []

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(6, 5), dpi=120)

    # -----------------------------
    # Time stepping
    # -----------------------------
    for step in tqdm(range(N_TIME_STEPS)):
        convection_x = velocity_x_prev * central_difference_x_periodic(velocity_x_prev)
        diffusion_x = KINEMATIC_VISCOSITY * laplace_periodic(velocity_x_prev)

        velocity_x_next = (
            velocity_x_prev
            +
            TIME_STEP_LENGTH
            * (
                -PRESSURE_GRADIENT[0]  # constant driving force in +x
                +
                diffusion_x
                -
                convection_x
            )
        )

        # Enforce wall BCs (no-slip)
        velocity_x_next[0, :] = 0.0
        velocity_x_next[-1, :] = 0.0

        # Diagnostics
        diff = velocity_x_next - velocity_x_prev
        residual = np.linalg.norm(diff) / np.sqrt(velocity_x_prev.size)
        residual_history.append(residual)

        # Sample centerline velocity at channel center (y≈0.5, x≈0.5)
        j_center = N_POINTS // 2
        i_center = N_POINTS // 2
        centerline_history.append(velocity_x_next[i_center, j_center])

        # Advance in time
        velocity_x_prev = velocity_x_next

        # -----------------------------
        # Live visualization (throttled)
        # -----------------------------
        if step % PLOT_EVERY == 0 or step == N_TIME_STEPS - 1:
            plt.clf()

            # Contour of u-velocity
            contour = plt.contourf(
                coordinates_x,
                coordinates_y,
                velocity_x_next,
                levels=30,
                cmap="viridis"
            )
            plt.colorbar(contour, label="u-velocity")

            # Quiver (can be downsampled if grid is large)
            plt.quiver(
                coordinates_x[::QUIVER_SKIP, ::QUIVER_SKIP],
                coordinates_y[::QUIVER_SKIP, ::QUIVER_SKIP],
                velocity_x_next[::QUIVER_SKIP, ::QUIVER_SKIP],
                np.zeros_like(velocity_x_next[::QUIVER_SKIP, ::QUIVER_SKIP]),
                color="white",
                scale=50
            )

            plt.xlabel("Position along pipe (x)")
            plt.ylabel("Position across pipe (y)")
            plt.title(f"Pressure-driven pipe flow – step {step}")

            # Velocity profile at a slice in x (e.g. column 1)
            ax_top = plt.gca().twiny()
            ax_top.plot(velocity_x_next[:, 1], y_range, color="white", lw=1.5)
            ax_top.set_xlabel("u(y) at x ≈ 0")

            plt.tight_layout()
            plt.pause(0.01)

    # -----------------------------
    # Final analysis plots (nicer)
    # -----------------------------
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=120)
    fig.suptitle("Pressure-Driven Pipe Flow – Final State & Diagnostics", fontsize=14)

    # 1) Final velocity field
    ax = axs[0, 0]
    c = ax.contourf(
        coordinates_x,
        coordinates_y,
        velocity_x_next,
        levels=30,
        cmap="viridis"
    )
    fig.colorbar(c, ax=ax, label="u-velocity")
    ax.quiver(
        coordinates_x[::QUIVER_SKIP, ::QUIVER_SKIP],
        coordinates_y[::QUIVER_SKIP, ::QUIVER_SKIP],
        velocity_x_next[::QUIVER_SKIP, ::QUIVER_SKIP],
        np.zeros_like(velocity_x_next[::QUIVER_SKIP, ::QUIVER_SKIP]),
        color="black",
        scale=50
    )
    ax.set_title("Final u-velocity field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # 2) Final velocity profile vs. analytical Hagen–Poiseuille
    ax = axs[0, 1]
    u_avg_y = velocity_x_next.mean(axis=1)  # average in x
    y = y_range

    # Analytical parabolic profile: u(y) = -(dp/dx) / (2ν) * y (1 - y)
    dpdx = PRESSURE_GRADIENT[0]  # here negative
    u_analytical = -dpdx * y * (1.0 - y) / (2.0 * KINEMATIC_VISCOSITY)

    ax.plot(u_avg_y, y, "o-", label="Simulated (x-avg)", color="deepskyblue")
    ax.plot(u_analytical, y, "r--", label="Analytical Hagen–Poiseuille")
    ax.set_title("Velocity profile u(y)")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # 3) Residual vs. time step
    ax = axs[1, 0]
    steps = np.arange(N_TIME_STEPS)
    ax.semilogy(steps, residual_history, "-o", markersize=3, color="orange")
    ax.set_title("Residual vs. iteration")
    ax.set_xlabel("time step")
    ax.set_ylabel("L2 residual (u^{n+1} - u^n)")
    ax.grid(alpha=0.3)

    # 4) Centerline velocity vs. time step
    ax = axs[1, 1]
    ax.plot(steps, centerline_history, "-o", markersize=3, color="lime")
    ax.set_title("Centerline velocity vs. iteration")
    ax.set_xlabel("time step")
    ax.set_ylabel("u(center)")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
