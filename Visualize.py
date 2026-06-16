import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch



def plotting(
    profits_log,
    price_log,
    invest_log,
    config,
    save_path="TrainingResults/training_plots.png",
    show=False,
    fig_size=(16, 8.125),
    dpi=300,
    max_plot_points=10000,
):

    logs = [
        ("Profits", "Profit", profits_log),
        ("Prices", "Price", price_log),
        ("Investments", "Investment", invest_log),
    ]

    matrices = []
    for title, _, log in logs:
        matrix = np.asarray(log)
        matrices.append(matrix)

    total_points = len(matrices[0])
    turns = np.arange(total_points)

    downsample_step = max(1, int(np.ceil(total_points / max_plot_points)))
    if downsample_step > 1:
        plot_turns = []
        plot_matrices = [[] for _ in matrices]

        for start in range(0, total_points, downsample_step):
            end = min(start + downsample_step, total_points)
            plot_turns.append(turns[end - 1])

            for matrix_index, matrix in enumerate(matrices):
                plot_matrices[matrix_index].append(matrix[start:end].mean(axis=0))

        plot_turns = np.asarray(plot_turns)
        plot_matrices = [np.asarray(plot_matrix) for plot_matrix in plot_matrices]
    else:
        plot_turns = turns
        plot_matrices = matrices

    smooth_window = max(1, len(plot_turns) // 20)
    label_x = plot_turns[min(smooth_window, len(plot_turns) - 1)]


    fig, axes = plt.subplots(3, 1, figsize=fig_size, sharex=True)
    firm_colors = ["tab:blue", "tab:orange"]

    for ax, (title, ylabel, _), plot_matrix in zip(axes, logs, plot_matrices):

        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            smooth_firm_1 = np.convolve(plot_matrix[:, 0], kernel, mode='valid')
            smooth_firm_2 = np.convolve(plot_matrix[:, 1], kernel, mode='valid')
            smooth_turns = plot_turns[smooth_window - 1:]

            ax.plot(smooth_turns, smooth_firm_1, linewidth=1, color=firm_colors[0], label=f'Firm 1 trend ({smooth_window})')
            ax.plot(smooth_turns, smooth_firm_2, linewidth=1, color=firm_colors[1], label=f'Firm 2 trend ({smooth_window})')

        ax.set_ylabel(ylabel)
        ax.set_title(f'Firm 1 vs Firm 2 {title.lower()}')

        monopoly_colour = '#C62828'
        leader_colour = '#1565C0'
        follower_colour = '#2E7D32'

        if title.lower() == "prices":
            width = 1.2
            size = 8
            #Monopoly price line
            monopoly_y = config.MonopolyP 
            ax.axhline(y=monopoly_y, color=monopoly_colour, linestyle='-', linewidth=width)
            ax.text(x=label_x,y=monopoly_y,s='*Monopoly',color=monopoly_colour,va='center',ha='left',fontsize=size,
                    bbox=dict(
                        facecolor='white',    
                        boxstyle='round,pad=0.2' 
                    )
                    )
            
            if config.firms !=1:

                #Leader price
                leader_y = config.LeaderP 
                ax.axhline(y=leader_y, color=leader_colour, linestyle='-', linewidth=width)
                ax.text(x=label_x,y=leader_y,s='*Leader',color=leader_colour,va='center',ha='left',fontsize=size,
                        bbox=dict(
                            facecolor='white',    
                            boxstyle='round,pad=0.2' 
                        )
                        )

                #Follower price
                follower_y = config.FollowerP 
                ax.axhline(y=follower_y, color=follower_colour, linestyle='-', linewidth=width)
                ax.text(x=label_x,y=follower_y,s='*Follower',color=follower_colour,va='center',ha='left',fontsize=size,
                        bbox=dict(
                            facecolor='white',    
                            boxstyle='round,pad=0.2' 
                        )
                        )
                
        elif title.lower() == "investments":
            width = 1.2
            size = 8
            #Monopoly price line
            monopoly_y = config.MonopolyX 
            ax.axhline(y=monopoly_y, color=monopoly_colour, linestyle='-', linewidth=width)
            ax.text(x=label_x,y=monopoly_y,s='*Monopoly',color=monopoly_colour,va='center',ha='left',fontsize=size,
                    bbox=dict(
                        facecolor='white',    
                        boxstyle='round,pad=0.2' 
                    )
                    )
            
            if config.firms !=1:

                #Leader price
                leader_y = config.LeaderX 
                ax.axhline(y=leader_y, color=leader_colour, linestyle='-', linewidth=width)
                ax.text(x=label_x,y=leader_y,s='*Leader',color=leader_colour,va='center',ha='left',fontsize=size,
                        bbox=dict(
                            facecolor='white',    
                            boxstyle='round,pad=0.2' 
                        )
                        )

                #Follower price
                follower_y = config.FollowerX
                ax.axhline(y=follower_y, color=follower_colour, linestyle='-', linewidth=width)
                ax.text(x=label_x,y=follower_y,s='*Follower',color=follower_colour,va='center',ha='left',fontsize=size,
                        bbox=dict(
                            facecolor='white',    
                            boxstyle='round,pad=0.2' 
                        )
                        )
            
        elif title.lower() == "profits":
            width = 1.2
            size = 8
            #Monopoly price line
            monopoly_y = config.MonopolyProfit
            ax.axhline(y=monopoly_y, color=monopoly_colour, linestyle='-', linewidth=width)
            ax.text(x=label_x,y=monopoly_y,s='*Monopoly',color=monopoly_colour,va='center',ha='left',fontsize=size,
                    bbox=dict(
                        facecolor='white',    
                        boxstyle='round,pad=0.2' 
                    )
                    )
            
            if config.firms !=1:

                #Leader price
                leader_y = config.LeaderProfit 
                ax.axhline(y=leader_y, color=leader_colour, linestyle='-', linewidth=width)
                ax.text(x=label_x,y=leader_y,s='*Leader',color=leader_colour,va='center',ha='left',fontsize=size,
                        bbox=dict(
                            facecolor='white',    
                            boxstyle='round,pad=0.2' 
                        )
                        )

                #Follower price
                follower_y = config.FollowerProfit
                ax.axhline(y=follower_y, color=follower_colour, linestyle='-', linewidth=width)
                ax.text(x=label_x,y=follower_y,s='*Follower',color=follower_colour,va='center',ha='left',fontsize=size,
                        bbox=dict(
                            facecolor='white',    
                            boxstyle='round,pad=0.2' 
                        )
                        )


        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    axes[-1].set_xlabel('Turns')
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))


    handles, labels = ax.get_legend_handles_labels()

    # Place a single legend below all three plots.
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(labels))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_visit_counts_3d(
    firm1_counts,
    firm2_counts,
    save_path="TrainingResults/visit_counts_3d.png",
    show=False,
    fig_size=(16, 8.125),
    dpi=300,
):
    count_matrices = [
        ("Firm 1 state-action decisions", np.asarray(firm1_counts)),
        ("Firm 2 state-action decisions", np.asarray(firm2_counts)),
    ]

    fig = plt.figure(figsize=fig_size,constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.08,
        h_pad=0.08,
        wspace=0.10,
        hspace=0.05
    )


    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]

    for ax, (title, counts) in zip(axes, count_matrices):
        if counts.ndim < 2:
            raise ValueError("Visit count matrices must include at least one state dimension and one action dimension")

        action_count = counts.shape[-1]
        leader_axis = counts.ndim - 2
        if counts.shape[leader_axis] < 2:
            raise ValueError("Visit count matrices must have leader as the last state dimension with values 0 and 1")


   
  

        action_indices = np.arange(action_count)
        state_offset = 0
        surfaces = []

        for leader_value, cmap in [(1, "viridis"), (0, "plasma")]:
            leader_counts = np.take(counts, leader_value, axis=leader_axis)
            plot_counts = leader_counts.reshape(-1, action_count)
            state_indices = np.arange(state_offset, state_offset + len(plot_counts))
            state_grid, action_grid = np.meshgrid(state_indices, action_indices)
            visit_grid = plot_counts.T
            visit_grid_from_floor = visit_grid

            if leader_value == 1:
                Position_text = "Leader"
            else:
                Position_text = "Follower"

            if np.any(visit_grid):
                surface = ax.plot_surface(
                    state_grid,
                    action_grid,
                    visit_grid_from_floor,
                    cmap=cmap,
                    linewidth=0,
                    antialiased=False,
                    alpha=0.95,
                )
                surfaces.append((surface,Position_text))

     

            if leader_value == 1:
                ax.plot(
                    [state_offset + len(plot_counts) - 0.5] * 2,
                    [0, action_count - 1],
                    [0, 0],
                    color="black",
                    linewidth=1,
                    alpha=0.65,
                )

            state_offset += len(plot_counts)

      
        legend_handles = [
            Patch(facecolor=plt.cm.viridis(0.7), label="Leader"),
            Patch(facecolor=plt.cm.plasma(0.7), label="Follower"),
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.02)
        )




        ax.set_title(title)
        ax.set_xlabel("Grouped flattened state index")
        ax.set_ylabel("Action index")
        ax.set_zlim(bottom=0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path
