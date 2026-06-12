import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator


def plotting(
    profits_log,
    price_log,
    invest_log,
    max_points=1500,
    smooth_window=None,
    save_path="TrainingResults/training_plots.png",
    show=False,
    fig_size=(16, 8.125),
    dpi=120,
):

    logs = [
        ("Profits", "Profit", profits_log),
        ("Prices", "Price", price_log),
        ("Investments", "Investment", invest_log),
    ]

    matrices = []
    for title, _, log in logs:
        matrix = np.asarray(log)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError(f"{title} log must contain one firm pair per turn, e.g. [[firm1, firm2], ...]")
        matrices.append(matrix)

    total_points = len(matrices[0])
    if any(len(matrix) != total_points for matrix in matrices):
        raise ValueError("profits_log, price_log, and invest_log must have the same number of turns")

    turns = np.arange(total_points)

    if total_points > max_points:
        sample_idx = np.linspace(0, total_points - 1, max_points, dtype=int)
        plot_turns = turns[sample_idx]
        plot_matrices = [matrix[sample_idx] for matrix in matrices]
    else:
        plot_turns = turns
        plot_matrices = matrices

    if smooth_window is None:
        smooth_window = max(1, len(plot_turns) // 20)
    else:
        smooth_window = min(smooth_window, len(plot_turns))

    fig, axes = plt.subplots(3, 1, figsize=fig_size, sharex=True, constrained_layout=True)
    firm_colors = ["tab:blue", "tab:orange"]

    for ax, (title, ylabel, _), plot_matrix in zip(axes, logs, plot_matrices):
        ax.plot(plot_turns, plot_matrix[:, 0], linestyle='None', marker='.', markersize=2.5, alpha=0.25, color=firm_colors[0], label='Firm 1 points')
        ax.plot(plot_turns, plot_matrix[:, 1], linestyle='None', marker='.', markersize=2.5, alpha=0.25, color=firm_colors[1], label='Firm 2 points')

        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            smooth_firm_1 = np.convolve(plot_matrix[:, 0], kernel, mode='valid')
            smooth_firm_2 = np.convolve(plot_matrix[:, 1], kernel, mode='valid')
            smooth_turns = plot_turns[smooth_window - 1:]

            ax.plot(smooth_turns, smooth_firm_1, linewidth=0.5, color=firm_colors[0], label=f'Firm 1 trend ({smooth_window})')
            ax.plot(smooth_turns, smooth_firm_2, linewidth=0.5, color=firm_colors[1], label=f'Firm 2 trend ({smooth_window})')

        ax.set_ylabel(ylabel)
        ax.set_title(f'Firm 1 vs Firm 2 {title.lower()}')
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)
        ax.legend(loc="best")

    axes[-1].set_xlabel('Turns')
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_visit_counts_3d(
    firm1_counts,
    firm2_counts,
    max_points=10000,
    save_path="TrainingResults/visit_counts_3d.png",
    show=False,
    fig_size=(16, 8.125),
    dpi=120,
):
    count_matrices = [
        ("Firm 1 visit counts", np.asarray(firm1_counts)),
        ("Firm 2 visit counts", np.asarray(firm2_counts)),
    ]

    fig = plt.figure(figsize=fig_size, constrained_layout=True)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]

    for ax, (title, counts) in zip(axes, count_matrices):
        if counts.ndim < 2:
            raise ValueError("Visit count matrices must include at least one state dimension and one action dimension")

        flat_counts = counts.reshape(-1, counts.shape[-1])
        state_count, action_count = flat_counts.shape
        max_state_points = max(1, max_points // action_count)

        if state_count > max_state_points:
            state_indices = np.linspace(0, state_count - 1, max_state_points, dtype=int)
            plot_counts = flat_counts[state_indices]
        else:
            state_indices = np.arange(state_count)
            plot_counts = flat_counts

        action_indices = np.arange(action_count)
        state_grid, action_grid = np.meshgrid(state_indices, action_indices)
        visit_grid = plot_counts.T

        if np.any(visit_grid):
            surface = ax.plot_surface(
                state_grid,
                action_grid,
                visit_grid,
                cmap="viridis",
                linewidth=0,
                antialiased=False,
                alpha=0.95,
            )
            fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08, label="Visits")
        else:
            ax.text2D(0.35, 0.5, "No visits recorded", transform=ax.transAxes)

        ax.set_title(title)
        ax.set_xlabel("Flattened state index")
        ax.set_ylabel("Action index")
        ax.set_zlabel("Visits")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path
