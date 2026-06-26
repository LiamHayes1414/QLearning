import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from Helper import block_average_2d,find_pattern
import networkx as nx
import matplotlib.patheffects as patheffects


def add_equilibrium_lines(ax, title, config, label_x,linewidth=1.2,size=8,borderWidth=0.5):
    monopoly_colour = '#C62828'
    leader_colour = '#1565C0'
    follower_colour = '#2E7D32'

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
        ax.axhline(y=y_value, color=colour, linestyle='-', linewidth=linewidth,zorder=1)
        ax.text(
            x=label_x,
            y=y_value,
            s=label,
            color=colour,
            va='center',
            ha='left',
            fontsize=size,
            zorder=3,
            bbox=dict(
                facecolor='white',
                boxstyle='round,pad=0.2',
                linewidth=borderWidth,
            )
        )

def plotting(
    profits_logs,
    price_logs,
    invest_logs,
    config,
    downsample,
    stat_log_counter,
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

    experimentation = exp_points - stat_log_counter
    exp_turns = np.arange(experimentation) * downsample
    stat_turns = np.arange(stat_log_counter) + max(exp_turns) +1

    total_turns = np.concatenate([exp_turns, stat_turns])

    smooth_window = max(1, exp_points // 1000)
    label_x = exp_turns[min(smooth_window, exp_points - 1)]

    fig, axes = plt.subplots(4, 1, figsize=fig_size, sharex=True,gridspec_kw={'height_ratios': [1, 1, 1, 0.25]})
    firm_colors = plt.cm.tab10.colors

    for metric_index, (ax, (title, ylabel, _,_), matrix_e,matrix_s) in enumerate(zip(axes[:3], logs, exp_matrices,stat_matrices)):

        kernel = np.ones(smooth_window) / smooth_window
        smooth_turns = total_turns[smooth_window - 1:]
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
                    stat_turns[-100:],
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

    y_vals = np.maximum((config.epsilon_decay*total_turns) + 1,0)

    # Plot the simple function
    ax4.plot(total_turns, y_vals, color='purple')
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
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

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
                    smooth_firm = np.convolve(leader_exp_matrix[:, firm_index],kernel,mode='valid')
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
            block_size = 1 if counts.shape[leader_axis] == 1 else len(action_indices)//2
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
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path

def strategy(price_statlog, invest_statlog, config, save_path="TrainingResults/strategy.png", show=False, dpi=300):
    print('Strategy')

    prices = np.asarray(price_statlog)
    investments = np.asarray(invest_statlog)
    num_firms = config.firms
 
    label_x = 0  # x-coordinate for labels
    Titlesize = 13
    LabelSize = 11
    LabelWidth = 1
    LabelLength = 5
    
    # Find pattern
    Price_Relationships,Price_States = find_pattern(prices)
    Investment_Relationships,Investment_States = find_pattern(investments)

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(16, 11), dpi=dpi,constrained_layout=True,gridspec_kw={'height_ratios': [1, 1, 0.25]})
    ax_price_map = axs[0, 0]
    ax_invest_map = axs[0, 1]
    ax_price_actions = axs[1, 0]
    ax_invest_actions = axs[1, 1]
    ax_price_lead = axs[2,0]
    ax_invest_lead = axs[2,1]

    #Place note at bottom of graph
    note_text = (
        "Relationship Charts: Show market state-action pairs.\n"
        "Market States: Separates history into distinct Pricing vs. Investment states for visualization. "
        "*Note: This differs from the main algorithm, where firms make decisions using only the pricing state.\n"
        "Anomalies (if present): Indicated by dashed lines on relationship charts and diamonds on action charts."
    )

    # Place text at x=0.02 (slightly off the left edge) and y=0.01 (very bottom edge)
    fig.text(
        0.02, 0.01, note_text, 
        fontsize=10, 
        color='gray', 
        style='italic',
        verticalalignment='bottom', 
        horizontalalignment='left'
    )

    #leave room at the bottom for the text
    fig.get_layout_engine().set(rect=[0, 0.05, 1, 1]) 

    def strategy_map(relationships, ax, title, node_color):
        ax.clear()
        G = nx.DiGraph()
        max_count = None
        min_count = None
        for source, target, count in relationships:
            G.add_edge(source, target, weight=count)

            #Update max count
            if max_count:
                if count>max_count:max_count = count
            else:
                max_count = count

            #Update min count
            if min_count:
                if count<min_count:min_count = count
            else:
                min_count = count

        if len(G.nodes()) > 1:
            k = 100 / np.sqrt(len(G.nodes()))
            pos = nx.spring_layout(G, k=k, iterations=1000, seed=42, scale=10)
            #pos = nx.kamada_kawai_layout(G, scale=3)
        else:
            pos = nx.spring_layout(G, seed=42)
        ax.set_title(title,fontsize=Titlesize)

        #Identify the max outgoing edge for each node
        max_edges = set()
        for node in G.nodes():
            outgoing = G.out_edges(node, data=True)
            if outgoing:
                max_edge = max(outgoing, key=lambda x: x[2].get("weight", 0))
                max_edges.add((max_edge[0], max_edge[1]))

        #Seperate primary and secondary lines
        primary_edges = []
        secondary_edges = []
        for u, v in G.edges():
            if (u, v) in max_edges:
                primary_edges.append((u, v))
            else:
                secondary_edges.append((u, v))

        #custom text labels for each node
        node_labels = {node: f"State {node}" for node in G.nodes()}

        #Draw the graph components
        Node_Size =  5000*(0.95 **(len(G.nodes())-1))
        Node_Radius = np.sqrt(Node_Size) / 2
        Label_Size = Node_Radius / 3
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=Node_Size, edgecolors="black",linewidths=0.75)
        nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=Label_Size, font_weight="bold")
        ax.margins(0.2)

        #line settings
        primary_color = 'black'
        primary_style = 'solid'

        secondary_color = 'grey'
        secondary_style = 'dashed'

        count_range = max_count - min_count
        min_line_size = 3
        max_line_size = 3
        line_size_range = max_line_size - min_line_size

        #Main Line
        primary_connections= []
        primary_width = []
        primary_arrowsz = []
        for u, v in primary_edges:
            #Scaling line sizes dynamically by count
            edge_weight = G[u][v]['weight']
            size_scale = 1 if count_range == 0 else (edge_weight - min_count)/count_range

            line_size = min_line_size + line_size_range*size_scale
            primary_width.append(line_size)
            primary_arrowsz.append(line_size+5)

            #Different connection style if arrow point to self
            if u == v:
                primary_connections.append("arc3,rad=0.0")
            else:
                primary_connections.append("arc3,rad=-0.1")
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=primary_edges,
            width=primary_width,
            edge_color=primary_color,
            style=primary_style,
            arrows=True,
            arrowsize=primary_arrowsz,
            arrowstyle="-|>",
            min_source_margin=Node_Radius,
            min_target_margin=Node_Radius,
            connectionstyle=primary_connections
        )
        #Secondary line
        secondary_connections = []
        secondary_width = []
        secondary_arrowsz = []
        for u, v in secondary_edges:
            edge_weight = G[u][v]['weight']
            size_scale = 1 if count_range == 0 else (edge_weight - min_count)/count_range

            line_size = min_line_size + line_size_range*size_scale
            secondary_width.append(line_size)
            secondary_arrowsz.append(line_size+5)
            if u == v:
                secondary_connections.append("arc3,rad=0.0")
            else:
                secondary_connections.append("arc3,rad=0.5")
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=secondary_edges,
            width=secondary_width,
            edge_color=secondary_color,
            style=secondary_style,
            arrows=True,
            arrowsize=secondary_arrowsz,
            arrowstyle="-|>",
            min_source_margin=Node_Radius,
            min_target_margin=Node_Radius,
            connectionstyle=secondary_connections
            )      

    def actions_display(Relationships,States,ax,title):
        ax.clear()
        Main_path = {
            first_idx: max((x for x in Relationships if x[0] == first_idx), key=lambda x: x[2])
            for first_idx in set(x[0] for x in Relationships)
        }
        secondary_paths = [x for x in Relationships if Main_path.get(x[0]) != x]

        State_path = [0]
        for _ in range(len(Main_path)):
            curr_state = State_path[-1]
            next_state = Main_path[curr_state][1]
            State_path.append(next_state)
        
        Filled_Main_Path =  States[State_path]
        state_labels = [
            f"State {num}'" if i == len(State_path) - 1 else f"State {num}" 
            for i, num in enumerate(State_path)
        ]
        #collect anomolies
        Anomolies = []
        for i in State_path[:-1]:
            from_i = [x[1] for x in secondary_paths if x[0] == i]

            for action in from_i:
                next_state = State_path[State_path.index(i)+1]
                Anomolies.append((next_state,action))

        Anomoly_XY = [(u, States[v, :-1]) for u, v in Anomolies]

        #Main line
        for i in range(Filled_Main_Path.shape[1] - 1):
            line, =ax.plot(state_labels,Filled_Main_Path[:, i], label=f"Firm {i+1}", marker='o',linewidth=2,markersize=3)
            current_line_color = line.get_color() #use same color for firm scatter points

            firm_average = np.mean(Filled_Main_Path[:, i])
            x_pos = state_labels[-1] 
            ax.plot(x_pos, firm_average, marker='<', markersize=8, color=current_line_color)
            ax.annotate(f'Avg:{firm_average:.2f}',xy=(x_pos, firm_average),xytext=(10, 0),textcoords='offset points',va='center',color=current_line_color,fontweight='light',
                        bbox=dict(
                            facecolor='white', 
                            edgecolor=current_line_color, 
                            boxstyle='round,pad=0.3'                  
                        ))
    

            #Extract anomoly points
            for tup in Anomoly_XY:
                val = tup[1][i]
                x_coord = f"State {tup[0]}'"if tup[0] == 0 else f"State {tup[0]}" 
                ax.scatter(x_coord,val,facecolors=current_line_color,edgecolors='black',marker="D", s=15, zorder=3,linewidths=1)

        if title == 'Price Actions':
            Eq_Type = 'prices'
        elif title == 'Investment Actions':
            Eq_Type = 'investments'
        else:
            raise ValueError('Strategy title does not match')

        add_equilibrium_lines(ax, Eq_Type, config, label_x,size=11,linewidth=2)

        ax.set_title(title, fontsize=Titlesize)
        ax.tick_params(axis='both', labelsize=LabelSize,length=LabelLength, width=LabelWidth,labelcolor='black')
        ax.set_ylabel("Price/Investment", fontsize=LabelSize)
        ax.tick_params(axis='x', labelrotation=45) 

        return Filled_Main_Path,State_path
    
    def leadership_display(MainPath,Index_Path,firms,ax,title):
        firm_labels = list(range(firms))
        state_labels = [
            f"State {num}'" if i == len(Index_Path) - 1 else f"State {num}" 
            for i, num in enumerate(Index_Path)
        ]

        Leader = MainPath[:,-1]
        ax.scatter(state_labels,Leader, color="black", marker="o", s=25)
        ax.set_yticks(firm_labels)
        ax.set_yticklabels([f"Firm {f+1}" for f in firm_labels], fontsize=LabelSize)
        padding = 0.5 
        ax.set_ylim(min(firm_labels) - padding, max(firm_labels) + padding)
        ax.set_title(title, fontsize=Titlesize)
        ax.tick_params(axis='both', labelsize=LabelSize, length=LabelLength, width=LabelWidth,labelcolor='black')
        ax.tick_params(axis='x', labelrotation=45) 

    # Run the functions for both pricing and investment
    strategy_map(Price_Relationships, ax_price_map, "Price Relationships", "lightblue")
    strategy_map(Investment_Relationships, ax_invest_map, "Investment Relationships", "lightgreen")

    Price_Path,Price_Path_Indexes = actions_display(Price_Relationships,Price_States,ax_price_actions,"Price Actions")
    Invest_Path,Invest_Path_Indexes = actions_display(Investment_Relationships,Investment_States,ax_invest_actions,"Investment Actions")
    
    leadership_display(Price_Path,Price_Path_Indexes,num_firms,ax_price_lead,"Market Leader - Price Pattern")
    leadership_display(Invest_Path,Invest_Path_Indexes,num_firms,ax_invest_lead,"Market Leader - Investment Pattern")

    for ax in axs.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(LabelWidth)

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

