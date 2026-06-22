import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from Helper import block_average_2d,find_pattern

def add_equilibrium_lines(ax, title, config, label_x):
    monopoly_colour = '#C62828'
    leader_colour = '#1565C0'
    follower_colour = '#2E7D32'
    width = 1.2
    size = 8

    if title.lower() == "prices":
        reference_values = [("*Monopoly", config.MonopolyP, monopoly_colour)]
        if config.firms > 1:
            reference_values.extend([
                ("*Leader", config.LeaderP, leader_colour),
                ("*Follower", config.FollowerP, follower_colour),
            ])
    elif title.lower() == "investments":
        reference_values = [("*Monopoly", config.MonopolyX, monopoly_colour)]
        if config.firms > 1:
            reference_values.extend([
                ("*Leader", config.LeaderX, leader_colour),
                ("*Follower", config.FollowerX, follower_colour),
            ])
    elif title.lower() == "profits":
        reference_values = [("*Monopoly", config.MonopolyProfit, monopoly_colour)]
        if config.firms > 1:
            reference_values.extend([
                ("*Leader", config.LeaderProfit, leader_colour),
                ("*Follower", config.FollowerProfit, follower_colour),
            ])
    else:
        reference_values = []

    for label, y_value, colour in reference_values:
        ax.axhline(y=y_value, color=colour, linestyle='-', linewidth=width)
        ax.text(
            x=label_x,
            y=y_value,
            s=label,
            color=colour,
            va='center',
            ha='left',
            fontsize=size,
            bbox=dict(
                facecolor='white',
                boxstyle='round,pad=0.2'
            )
        )

def plotting(
    profits_logs,
    price_logs,
    invest_logs,
    config,
    downsample,
    save_path="TrainingResults/training_plots.png",
    show=False,
    fig_size=(16, 11),
    dpi=300
):
    print('Plotting')
    profits_explog,profits_statlog = profits_logs
    price_explog, price_statlog = price_logs
    invest_explog,invest_statlog = invest_logs
 

    logs = [
        ("Profits", "Profit", profits_explog,profits_statlog),
        ("Prices", "Price", price_explog,price_statlog),
        ("Investments", "Investment", invest_explog,invest_statlog),
    ]

    num_firms = config.firms

    exp_matrices = []
    stat_matrices = []
    for title, _, log_e,log_s in logs:
        exp_matrix = np.asarray(log_e)
        exp_matrices.append(exp_matrix)

        stat_matrix = np.asarray(log_s)
        stat_matrices.append(stat_matrix)

    exp_points = len(exp_matrices[0])
    exp_turns = np.arange(exp_points) * downsample

    stat_turns = np.arange(len(stat_matrices[0])) + exp_points* downsample

    smooth_window = max(1, exp_points // 50)
    label_x = exp_turns[min(smooth_window, exp_points - 1)]

    fig, axes = plt.subplots(4, 1, figsize=fig_size, sharex=True,gridspec_kw={'height_ratios': [1, 1, 1, 0.25]})
    firm_colors = plt.cm.tab10.colors

    for metric_index, (ax, (title, ylabel, _,_), matrix_e,matrix_s) in enumerate(zip(axes[:3], logs, exp_matrices,stat_matrices)):

        kernel = np.ones(smooth_window) / smooth_window
        smooth_turns = exp_turns[smooth_window - 1:]
        for firm_index in range(num_firms):
                smooth_firm = np.convolve(matrix_e[:, firm_index], kernel, mode='valid')
                firm_color = firm_colors[firm_index % len(firm_colors)]
                ax.plot(
                    smooth_turns,
                    smooth_firm,
                    linewidth=1,
                    color=firm_color,
                    label=f'Firm {firm_index + 1} trend ({smooth_window})'
                )
                #plot last 100 points to see if there is oscillation
                ax.scatter(
                    stat_turns[:100], #using first 100 indexes so plotting is nicer
                    matrix_s[:, firm_index][-100:], #showing last 100
                    color=firm_color,
                    s=10,
                    zorder=3
                )

        ax.set_ylabel(ylabel)
        ax.set_title(title)

        add_equilibrium_lines(ax, title, config, label_x)

        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    ax4 = axes[3]
    decay_axis = np.concatenate((exp_turns, stat_turns[:100]))
    y_vals = np.maximum((config.epsilon_decay*decay_axis) + 1,0)

    # Plot the simple function
    ax4.plot(decay_axis, y_vals, color='purple')
    ax4.set_title(r'$\text{Exploration Probability } \epsilon = MAX(\left(-\frac{1}{\text{ExploreLen}}\right) \times \text{Round} + 1$, 0)')
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.25)
    ax4.margins(x=0)

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


def leaderplots(
    profits_log,
    price_log,
    invest_log,
    config,
    downsample=100,
    save_path="TrainingResults/leader_plots.png",
    show=False,
    fig_size=None,
    dpi=300
):
    print('Leader Plot')
    profits_explog, profits_statlog = profits_log
    price_explog, price_statlog = price_log
    invest_explog, invest_statlog = invest_log

    logs = [
        ("Profits", "Profit", profits_explog, profits_statlog),
        ("Prices", "Price", price_explog, price_statlog),
        ("Investments", "Investment", invest_explog, invest_statlog),
    ]
    num_firms = config.firms

    exp_matrices = []
    stat_matrices = []
    for _, _, log_e, log_s in logs:
        exp_matrices.append(np.asarray(log_e))
        stat_matrices.append(np.asarray(log_s))

    exp_points = len(exp_matrices[0])
    exp_turns = np.arange(exp_points) * downsample
    stat_turns = np.arange(len(stat_matrices[0])) + exp_points * downsample

    exp_leader_column = exp_matrices[0][:, num_firms]
    stat_leader_column = stat_matrices[0][:, num_firms]
    smooth_window = max(1, exp_points // 50)

    if fig_size is None:
        fig_size = (max(16, 8 * num_firms), 11)

    fig, axes = plt.subplots(
        4,
        num_firms,
        figsize=fig_size,
        sharex='col',
        gridspec_kw={'height_ratios': [1, 1, 1, 0.25]}
    )

    if num_firms == 1:
        axes = axes.reshape(4, 1)

    firm_colors = plt.cm.tab10.colors

    for leader_index in range(num_firms):
        exp_leader_mask = exp_leader_column == leader_index
        stat_leader_mask = stat_leader_column == leader_index
        leader_exp_turns = exp_turns[exp_leader_mask]
        leader_stat_turns = stat_turns[stat_leader_mask]
        label_turns = np.concatenate((leader_exp_turns, leader_stat_turns))
        label_x = label_turns[0] if len(label_turns) else 0

        for ax, (title, ylabel, _, _), matrix_e, matrix_s in zip(
            axes[:3, leader_index],
            logs,
            exp_matrices,
            stat_matrices
        ):
            leader_exp_matrix = matrix_e[exp_leader_mask, :num_firms]
            leader_stat_matrix = matrix_s[stat_leader_mask, :num_firms]
            leader_smooth_window = min(smooth_window, len(leader_exp_turns))

            if leader_smooth_window > 0:
                kernel = np.ones(leader_smooth_window) / leader_smooth_window
                smooth_turns = leader_exp_turns[leader_smooth_window - 1:]

                for firm_index in range(num_firms):
                    smooth_firm = np.convolve(
                        leader_exp_matrix[:, firm_index],
                        kernel,
                        mode='valid'
                    )
                    firm_color = firm_colors[firm_index % len(firm_colors)]
                    ax.plot(
                        smooth_turns,
                        smooth_firm,
                        linewidth=1,
                        color=firm_color,
                        label=f'Firm {firm_index + 1} trend ({leader_smooth_window})'
                    )

            for firm_index in range(num_firms):
                firm_color = firm_colors[firm_index % len(firm_colors)]
                ax.scatter(
                    leader_stat_turns[:100],
                    leader_stat_matrix[:, firm_index][-100:],
                    color=firm_color,
                    s=10,
                    zorder=3
                )

            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} - Firm {leader_index + 1} Leader")
            add_equilibrium_lines(ax, title, config, label_x)
            ax.grid(True, alpha=0.25)
            ax.margins(x=0)

        ax4 = axes[3, leader_index]
        decay_axis = np.concatenate((leader_exp_turns, leader_stat_turns[:100]))
        if len(decay_axis):
            y_vals = np.maximum((config.epsilon_decay * decay_axis) + 1, 0)
            ax4.plot(decay_axis, y_vals, color='purple')

        ax4.set_title(r'$\text{Exploration Probability } \epsilon$')
        ax4.set_ylim(-0.1, 1.1)
        ax4.grid(True, alpha=0.25)
        ax4.margins(x=0)

        if len(label_turns) == 0:
            axes[0, leader_index].text(
                0.5,
                0.5,
                f"No rounds where Firm {leader_index + 1} was leader",
                transform=axes[0, leader_index].transAxes,
                ha="center",
                va="center",
            )

        if leader_index > 0:
            for row_index in range(4):
                axes[row_index, leader_index].set_ylabel("")

        axes[-1, leader_index].set_xlabel('Turns')
        axes[-1, leader_index].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(labels)
        )

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
    firms,
    save_path="TrainingResults/visit_counts_3d.png",
    show=False,
    fig_size=(16, 8.125),
    dpi=300,
):
    print('3D Plot')

    fig = plt.figure(figsize=fig_size,constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.08,
        h_pad=0.08,
        wspace=0.10,
        hspace=0.05
    )

    #store values for each firm dynamically
    count_matrices = []
    axes = []
    for i,f in enumerate(firms):
        Firm = (f"Firm {i+1} state-action decisions", np.asarray(f.visit_counts))
        count_matrices.append(Firm)

        axes.append(fig.add_subplot(1, len(firms), i+1, projection="3d"))
   
    for ax, (title, counts) in zip(axes, count_matrices):
        action_count = counts.shape[-1]
        leader_axis = counts.ndim - 2
        action_indices = np.arange(action_count)
        state_offset = 0
        surfaces = []

        if counts.shape[leader_axis] == 1:
            position_plots = [(0, "viridis", "Monopoly")]
        else:
            position_plots = [(1, "viridis", "Leader"), (0, "plasma", "Follower")]

        for leader_value, cmap, Position_text in position_plots:
            leader_counts = np.take(counts, leader_value, axis=leader_axis)
            plot_counts = leader_counts.reshape(-1, action_count)
            state_indices = np.arange(state_offset, state_offset + len(plot_counts))
            state_grid, action_grid = np.meshgrid(state_indices, action_indices)
            visit_grid = plot_counts.T
            
            #smooth matrix for faster plotting
            block_size = 1 if counts.shape[leader_axis] == 1 else len(action_indices)//4
            smooth_states = block_average_2d(state_grid, block_size)
            smooth_actions = block_average_2d(action_grid, block_size)
            smooth_visits = block_average_2d(visit_grid, block_size)
       
            if np.any(visit_grid):
                surface = ax.plot_surface(
                    smooth_states,
                    smooth_actions,
                    smooth_visits,
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

      
        if counts.shape[leader_axis] == 1:
            legend_handles = [
                Patch(facecolor=plt.cm.viridis(0.7), label="Monopoly")
            ]
        else:
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

def strategy(price_statlog, invest_statlog, config, save_path="TrainingResults/strategy.png", show=False, fig_size=(12, 9), dpi=300):
    print('Strategy')

    prices = np.asarray(price_statlog)
    investments = np.asarray(invest_statlog)
    num_firms = config.firms
 
    label_x = 0  # x-coordinate for labels
    firm_colors = plt.cm.tab10.colors
    
    # Find pattern
    Price_Patterns = find_pattern(prices)
    Investment_Patterns = find_pattern(investments)

    print(Price_Patterns)

    #Get Valid patterns so i can use max valid
    valid_pattern_lengths = [len(p) for p in Price_Patterns if p is not None]
    
    if not valid_pattern_lengths:  # If the list is empty (any pattern was None or all None)
        has_pattern = False
        View = 50
        pattern_len = 0
    else:
        has_pattern = True
        pattern_len = max(valid_pattern_lengths)
        View = int(pattern_len * 3) + 1

    view_points = min(View, len(prices))
    turns = np.arange(view_points)

    # Instantiate single canvas and axes cleanly
    fig, axes = plt.subplots(
        3,
        1,
        figsize=fig_size,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 0.6]}
    )
    
    # Grid ticks at every integer cycle step
    axes[0].set_xticks(turns) 
    logs = [
        ("Prices", prices, axes[0]),
        ("Investments", investments, axes[1]),
    ]

    for title, matrix, ax in logs:
        plot_matrix = matrix[-view_points:, :num_firms]
        for firm_index in range(num_firms):
            ax.plot(
                turns,
                plot_matrix[:, firm_index],
                color=firm_colors[firm_index % len(firm_colors)],
                label=f"Firm {firm_index + 1}"
            )
            
            # Draw vertical pattern dividers if an overarching cycle length was isolated
            if has_pattern:
                ax.axvspan(0, pattern_len, facecolor='none', alpha=0.2, linewidth=2, linestyle='--', edgecolor='black')
                ax.axvspan(pattern_len, pattern_len*2, facecolor='none', alpha=0.2, linewidth=2, linestyle='--', edgecolor='black')
                ax.axvspan(pattern_len*2, pattern_len*3, facecolor='none', alpha=0.2, linewidth=2, linestyle='--', edgecolor='black')

        add_equilibrium_lines(ax, title, config, label_x)
        ax.set_title(title)
        ax.set_ylabel(title[:-1])
        ax.grid(True, alpha=0.25)

    leader_values = prices[-view_points:, num_firms]
    
    axes[2].scatter(
        turns,
        leader_values,
        color="black",
        s=14,
        zorder=3
    )
    axes[2].set_title("Leader Index")
    axes[2].set_ylabel("Leader")
    
    # Configure Y-axis ticks to account for both active firm numbers and no leader state (if ever happens)
    y_ticks = np.append([-1], np.arange(num_firms))
    y_labels = ["None"] + [f"Firm {i+1}" for i in range(num_firms)]
    axes[2].set_yticks(y_ticks)
    axes[2].set_yticklabels(y_labels)
    axes[2].set_ylim(-1.5, num_firms - 0.5)
    axes[2].grid(True, alpha=0.25)

    axes[-1].set_xlabel("Stationary iteration")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=num_firms)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return Price_Patterns, Investment_Patterns
