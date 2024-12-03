import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_trajectories(
    paths,
    jumps=None,
    timesteps=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    title="CMB",
    N=2000,
    color="darkblue",
    cmap="tab10",
    show_paths=False,
):
    """Plot trajectories of some selected samples."""
    _, ax = plt.subplots(1, len(timesteps), figsize=(2.4 * len(timesteps), 2.75))

    for j, time in enumerate(timesteps):
        idx_path = int(time * len(paths)) if time < 1 else -1
        vmin = jumps.min() if jumps is not None else None
        vmax = jumps.max() if jumps is not None else None
        color = jumps[idx_path, :N] if jumps is not None else color

        if show_paths:
            for i in range(N):
                ax[j].plot(
                    paths[:idx_path, i, 0],
                    paths[:idx_path, i, 1],
                    alpha=0.3,
                    lw=0.1,
                    color="k",
                )  # Plot lines for each trajectory

        ax[j].scatter(
            paths[0, :N, 0],
            paths[0, :N, 1],
            s=1,
            color="gray",
            alpha=0.2,
            vmin=vmin,
            vmax=vmax,
        )
        ax[j].scatter(
            paths[idx_path, :N, 0],
            paths[idx_path, :N, 1],
            s=1,
            c=color,
            cmap=cmap,
            alpha=1,
            vmin=vmin,
            vmax=vmax,
        )
        ax[j].text(
            0.125,
            0.95,
            f"t={idx_path/len(paths) if time < 1 else 1:.1f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[j].transAxes,
            fontsize=10,
        )
        ax[j].set_xlim(-1.25, 1.25)
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].axis("equal")

    ax[0].set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def animate_trajectories(
    paths,
    jumps,
    title="CMB",
    N=2000,
    cmap="tab10",
    show_paths=True,
    filename="trajectories.gif",
    fps=10,
):
    """
    Create an animation of trajectories over time and save it as a GIF.
    """
    N = min(N, paths.shape[1])
    num_frames = paths.shape[0]
    jumps = jumps.squeeze(-1)  # Remove last dimension if it's of size 1

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.close()  # Prevents extra output in notebooks

    # Determine color mapping range
    vmin, vmax = jumps.min(), jumps.max()
    norm = BoundaryNorm(np.arange(vmin - 0.5, vmax + 1.5), vmax - vmin + 2)

    # Initialize the scatter plots
    scatter_start = ax.scatter([], [], s=2, color="gray", alpha=0.3, label="Start")
    scatter_current = ax.scatter([], [], s=2, cmap=cmap, norm=norm)

    # Optionally, initialize the trajectories
    if show_paths:
        lines = [ax.plot([], [], lw=0.5, alpha=0.3, color="k")[0] for _ in range(N)]

    # Set axis properties
    ax.set_xlim(paths[:, :N, 0].min(), paths[:, :N, 0].max())
    ax.set_ylim(paths[:, :N, 1].min(), paths[:, :N, 1].max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper left")
    ax.axis("equal")

    # Animation update function
    def update(frame):
        idx_path = frame

        # Update starting positions
        scatter_start.set_offsets(paths[0, :N])

        # Update current positions and colors
        scatter_current.set_offsets(paths[idx_path, :N])
        scatter_current.set_array(jumps[idx_path, :N])

        if show_paths:
            for i in range(N):
                # Update trajectories
                lines[i].set_data(
                    paths[: idx_path + 1, i, 0], paths[: idx_path + 1, i, 1]
                )

        return scatter_start, scatter_current, *(lines if show_paths else [])

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, blit=True, interval=1000 / fps
    )

    # Save the animation as a GIF
    ani.save(filename, writer="pillow", fps=fps)
    print(f"Animation saved as {filename}")
