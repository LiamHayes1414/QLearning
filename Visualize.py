import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from Helper import block_average_2d
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pathlib import Path
import pandas as pd
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import Normalize, ListedColormap

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
        reference_values = [("*Monopoly_L", config.MonopolyLeaderProfit, monopoly_colour)]
        if config.firms > 1:
            reference_values.extend([
                ("*Leader", config.LeaderProfit, leader_colour),
                ("*Follower", config.FollowerProfit, follower_colour),
                ("*Monopoly_F", config.MonopolyFollowerProfit, monopoly_colour),
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
    label_x = exp_turns[min(smooth_window, len(exp_turns) - 1)]

    fig, axes = plt.subplots(4, 1, figsize=fig_size, sharex=True,gridspec_kw={'height_ratios': [1, 1, 1, 0.25]})
    firm_colors = plt.cm.tab10.colors

    for metric_index, (ax, (title, ylabel, _,_), matrix_e,matrix_s) in enumerate(zip(axes[:3], logs, exp_matrices,stat_matrices)):

        kernel = np.ones(smooth_window) / smooth_window
        smooth_turns = total_turns[smooth_window - 1:]
        plot_start = smooth_turns[0]
        plot_end = smooth_turns[-1]
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
        ax.set_xlim(left=plot_start, right=plot_end)

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
    ax4.set_xlim(left=plot_start, right=plot_end)

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
    downsample,
    stat_log_counter,
    save_path="TrainingResults/leader_plots.png",
    show=False,
    fig_size=None,
    dpi=300
    ):
    print('Leader Plot')
    num_firms = config.firms
    if fig_size is None:
        fig_size = (max(16, 8 * num_firms), 11)

    fig, axes = plt.subplots(3,num_firms,figsize=fig_size,sharex='col',gridspec_kw={'height_ratios': [1, 1, 1]})

    if num_firms == 1:axes = axes.reshape(3, 1)

    firm_colors = plt.cm.tab10.colors
    
    profits_explog, profits_statlog = profits_log
    price_explog, price_statlog = price_log
    invest_explog, invest_statlog = invest_log

    logs = [
        ("Profits", "Profit", profits_explog, profits_statlog),
        ("Prices", "Price", price_explog, price_statlog),
        ("Investments", "Investment", invest_explog, invest_statlog),
    ]
    

    exp_matrices = []
    stat_matrices = []
    for _, _, log_e, log_s in logs: 
                                            #remove stationary values from exp_matrices (different appending frequency will be incorrect for leader filtering)
        exp_matrices.append(np.asarray(log_e)[:-stat_log_counter])
        stat_matrices.append(np.asarray(log_s))

    
    #See who is the market leader for each data point
    leader_column_exp = exp_matrices[0][:, num_firms]
    leader_column_stat = stat_matrices[0][:, num_firms]

    for leader_index in range(num_firms): #columns in diagram
        leader_mask_exp = leader_column_exp == leader_index
        leader_mask_stat = leader_column_stat == leader_index
        
        for ax, (title, ylabel, _, _), matrix_e, matrix_s in zip(axes[:3, leader_index],logs,exp_matrices,stat_matrices): #Rows in diagram
            leaderI_turns_exp = matrix_e[leader_mask_exp]
            leaderI_turns_stat = matrix_s[leader_mask_stat][:, :-1]
          
            leaderI_exp_points = len(leaderI_turns_exp)

            exp_turns = np.arange(leaderI_exp_points) * downsample
       
            total_turns = exp_turns
            total_points = leaderI_turns_exp

            last_x = total_turns[-1]

            smooth_window = max(1, leaderI_exp_points // 50)
            label_x = exp_turns[min(smooth_window, leaderI_exp_points - 1)]

            kernel = np.ones(smooth_window) / smooth_window
            smooth_turns = total_turns[smooth_window - 1:]
     
            for firm_index in range(num_firms):
                smooth_firm = np.convolve(total_points[:, firm_index], kernel, mode='valid')
                firm_color = firm_colors[firm_index % len(firm_colors)]
                ax.plot(
                    smooth_turns,
                    smooth_firm,
                    linewidth=1,
                    color=firm_color,
                    label=f'Firm {firm_index + 1} trend ({smooth_window})'
                )
            #plot stat points as box plot to see converged solutions
            n = leaderI_turns_stat.shape[1]
            
            right_padding = last_x*0.05

            box_width = 0.05 * last_x

            spacing = box_width * 1.5

            positions = last_x + box_width / 2 + np.arange(n) * spacing

            final_x = positions[-1] + box_width / 2 + right_padding

            bp = ax.boxplot(
                leaderI_turns_stat,
                positions=positions,
                widths=box_width,
                patch_artist=True
            )

            for firm_index, box in enumerate(bp['boxes']):
                firm_color = firm_colors[firm_index % len(firm_colors)]
                box.set_edgecolor(firm_color)
                box.set_facecolor('none')

                bp['whiskers'][2 * firm_index].set_color(firm_color)
                bp['whiskers'][2 * firm_index + 1].set_color(firm_color)
                # Caps (2 per box)
                bp['caps'][2 * firm_index].set_color(firm_color)
                bp['caps'][2 * firm_index + 1].set_color(firm_color)

                # Outliers (1 collection per box)
                bp['fliers'][firm_index].set_markeredgecolor(firm_color)
                bp['fliers'][firm_index].set_markerfacecolor('none')

                bp['medians'][firm_index].set_color(firm_color)

            ax.set_xlim(right=final_x)

            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} - Firm {leader_index + 1} Leader")
            add_equilibrium_lines(ax, title, config, label_x)
            ax.grid(True, alpha=0.25)
        
        #Hide y axis on all column except first one (save space)
        if leader_index > 0:
            for row_index in range(3):
                axes[row_index, leader_index].set_ylabel("")

        axes[-1, leader_index].set_xlabel('Turns')
        axes[-1, leader_index].set_xlim(0, final_x)

        ticks = np.linspace(0, last_x, 8)
        axes[-1, leader_index].set_xticks(ticks)
        axes[-1, leader_index].set_xticklabels([f"{x:,.0f}" for x in ticks])

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
            #block_size = 1 if counts.shape[leader_axis] == 1 else len(action_indices)//2
            #smooth_states = block_average_2d(state_grid, block_size)
            #smooth_actions = block_average_2d(action_grid, block_size)
            #smooth_visits = block_average_2d(visit_grid, block_size)
       
            # Plot all visit counts normally
            surface = ax.plot_surface(
                state_grid,
                action_grid,
                visit_grid,
                cmap=cmap,
                linewidth=0,
                antialiased=False,
                alpha=0.95,
            )

            # Highlight zero-visit cells in red
            zero_mask = visit_grid == 0

            if np.any(zero_mask):
                ax.scatter(
                    state_grid[zero_mask],
                    action_grid[zero_mask],
                    np.full(np.sum(zero_mask), 2000),
                    color="red",
                    s=12,
                    marker="s",
                    alpha=1.0,
                    depthshade=False,
                )
     

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
        ax.set_xlabel("Flattened state index")
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
        
def strategy(StateLogs,Firms,config, save_path="TrainingResults/StateEvolv.png", show=False, dpi=300):

    print('Strategy')
    num_firms = len(Firms)
    #Stores the possible actions for each scenario given current state
    Next_States = {}
    Next_States_IDs = {}
    StateMap ={state: str(i)for i, state in enumerate(StateLogs)}
    for state in StateLogs:
        StateID = StateMap[state]
        Possible_Actions= {i: [] for i in range(1, num_firms + 1)}
        for i,f in enumerate(Firms,start=1):
            Responses = f.Stat_Responses
            Price_index = f.decodelog(state)

            #see if the firm is ever the leader and follower for current state (specifically for no investments option, but also acts as a general safety)
            matches = [k for k in Responses if list(k[:len(Price_index)]) == Price_index]
            leadership_indicators = [t[-1] for t in matches]
    
            #possible states if leader or follower
            FollowerState = tuple(Price_index+[0])
            LeaderState = tuple(Price_index+[1])

            #Only include if leaderships status exists
            if 0 in leadership_indicators: FollowerResponse = Responses[FollowerState]
            if 1 in leadership_indicators: LeaderResponse = Responses[LeaderState]
                
            #Add responses to dict key in line with which firm is the potnetial new leader (leadership from market perspective not firm perspective)
            for key in Possible_Actions.keys():
                if key == i:
                    if config.firms==1: #Monopoly case
                        Possible_Actions[key].append(FollowerResponse)
                    else:
                        #only add leader response if it exists
                        if 1 in leadership_indicators: Possible_Actions[key].append(LeaderResponse)
                else:
                    #only add follower response if it exists
                    if 0 in leadership_indicators: Possible_Actions[key].append(FollowerResponse)

        Market_Actions =  {key: tuple(t[0] for t in value)for key, value in Possible_Actions.items()}
        
        Next_States[state] = {}
        Next_States_IDs[StateID] = {}

        for Leader, tup in Market_Actions.items():
            tup_sz = len(tup)
            if tup_sz ==0: #state does not exist
                continue

            #drop last prices and add new ones
            New_State = state[tup_sz:] + tup
       
            

            #Store pathways (numbers and IDs)
            Next_States[state][Leader] = New_State
            try:
                Next_States_IDs[StateID][Leader] = StateMap[New_State] 
            except:
                Next_States_IDs[StateID][Leader] ="\u03b8"
                print("**Non-visited State**",New_State)

    G = nx.MultiDiGraph()

    # Add edges
    for state, actions in Next_States_IDs.items():
        destinations = list(actions.values())

        # Check if all actions lead to the same state
        if len(set(destinations)) == 1:
            G.add_edge(
                state,
                destinations[0],
                label="combined"
            )
        else:
            for action, next_state in actions.items():
                G.add_edge(
                    state,
                    next_state,
                    label=action
                )

    # Position nodes
    G.graph.update({
        "ranksep": "3.0 equally",
        "nodesep": "1.5",
        "splines": "true",
        "node": {
            "width": "0.9",
            "height": "0.9",
            "fixedsize": "true",
        }
    })

    pos = graphviz_layout(G, prog="dot")
    

    fig = plt.figure(figsize=(14, 8))

    # Draw nodes and labels
    node_colors = [
        "tomato" if node == "\u03b8" else "skyblue"
        for node in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors)
    nx.draw_networkx_labels(G, pos,font_size=12)

    # Get unique labels
    labels = set(
        d["label"]
        for _, _, _, d in G.edges(data=True, keys=True)
        if d["label"] != "combined"
    )
    colors = cm.get_cmap("Set1", len(labels))

    # Draw edges by label category (which firm is the leader)
    for i, label in enumerate(sorted(labels)):

        edges = [
            (u, v, k)
            for u, v, k, d in G.edges(data=True, keys=True)
            if d["label"] == label
        ]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=[colors(i)],
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>',
            connectionstyle="arc3,rad=0.15",
            min_source_margin=25,
            min_target_margin=25
        )


    # Draw collapsed edges as dashed black arrows
    combined_edges = [
        (u, v, k)
        for u, v, k, d in G.edges(data=True, keys=True)
        if d["label"] == "combined"
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=combined_edges,
        edge_color="black",
        style="dashed",
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>',
        connectionstyle="arc3,rad=0.15",
        min_source_margin=25,
        min_target_margin=25
    )


    # Legend for actions
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors(i),
            lw=2,
            label=f"Firm {label} Leader"
        )
        for i, label in enumerate(sorted(labels))
    ]

    # Add collapsed transition legend
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            lw=2,
            linestyle="dashed",
            label="Same outcome"
        )
    )

   

    plt.legend(handles=legend_elements)

    plt.axis("off")

    plt.subplots_adjust(bottom=0.12)

    fig.text(
        x=0.5, 
        y=0.02, 
        s="Note: Node IDs represent visited pricing states. Node \u03b8 is hypothetically possible " \
            "but never visited because zero investment makes that leadership transition impossible.",
        ha="center",       
        va="bottom",       
        fontsize=8,       
        style="italic",    
        color="dimgray",
        wrap=True          # Prevents text cutting off horizontally
    )


    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def reactions(Firms,Possible_States,Possible_Actions,config):
    print("Reactions")

    def StateUpdate(firm,state,Prices):
        PrevStatePrices = list(state[:-1])
        PrevStatePrices.extend(Prices)
        del PrevStatePrices[:len(Prices)]
        state_index = firm.decodelog(PrevStatePrices)

        LeaderIndex = state_index.copy()
        FollowerIndex = state_index.copy()

        LeaderIndex.append(1)
        FollowerIndex.append(0)

        return LeaderIndex,FollowerIndex
    
    MainFirm = Firms[0]
    Price_Chosen = []
    Price_Optimal = []
    Investment_Chosen = []
    Investment_Optimal = []
    Deviation_Costs = []

    for state in Possible_States:
        #_______________Optimal actions for the given state (all firms)__________________
        OptimalInvestments = []
        OptimalPrices = []
        #Hard coded - since state is from view of firm 1, here for simplicity i assume that if firm 1 is not the leader then it is firm 2 - even if there are 3+ firms
        if state[-1] == 1:
            mrktleadr = 0
        else:
            mrktleadr = 1
            
        for i,firm in enumerate(Firms):
            if i == mrktleadr:
                Action = firm.Action(state[:-1], 0,Leadership=1) 
            else:
                Action = firm.Action(state[:-1], 0,Leadership=0)
            OptimalInvestments.append(Action[1])
            OptimalPrices.append(Action[0])
        MainPrice = OptimalPrices[0]
        MainInvestment = OptimalInvestments[0]

        #Based on optimal actions taken above, the next optimal state would be the following (either firm is leader or follower)
        LeaderIndex_opt,FollowerIndex_opt= StateUpdate(MainFirm,state,OptimalPrices)

        #In the next optimal state found in the line above, what are the best possible values I can achieve (leader or follower)
        LeaderValue_opt = max(MainFirm.Q_matrix[tuple(LeaderIndex_opt)])
        FollowerValue_opt = max(MainFirm.Q_matrix[tuple(FollowerIndex_opt)])

        #Probability that firm i will be leader in the next state
        if mrktleadr==0:
            LeadershipProb_opt = (OptimalInvestments[0]+config.K)/(sum(OptimalInvestments)+config.K)
        else:
            LeadershipProb_opt = (OptimalInvestments[0])/(sum(OptimalInvestments)+config.K)

        ExpectedValue_opt = (LeadershipProb_opt*LeaderValue_opt) + (FollowerValue_opt*(1-LeadershipProb_opt))

        #Save a copy of the optimal price actions for counter factual    
        ChosenPrices = OptimalPrices.copy()
        ChosenInvestments = OptimalInvestments.copy() 

        for action in Possible_Actions:
            #New hypothetical pricing and investment
            ChosenPrices[0] = action[0]
            ChosenInvestments[0] = action[1]

            #hypothetical state
            LeaderIndex_hyp,FollowerIndex_hyp= StateUpdate(MainFirm,state,ChosenPrices)

            #hypothetical values (leader or follower)
            LeaderValue_hyp = max(MainFirm.Q_matrix[tuple(LeaderIndex_hyp)])
            FollowerValue_hyp = max(MainFirm.Q_matrix[tuple(FollowerIndex_hyp)])

            #Calculate Probability that firm i will be leader
            if mrktleadr==0:
                LeadershipProb_hyp = (action[1]+config.K)/(sum(OptimalInvestments)+config.K)
            else:
                LeadershipProb_hyp = (action[1])/(sum(OptimalInvestments)+config.K)
            """^Leader has additional chance to remain leader if no innovation occurs (+k)"""

            ExpectedValue_hyp = (LeadershipProb_hyp*LeaderValue_hyp) + (FollowerValue_hyp*(1-LeadershipProb_hyp))

            DeviationCost = ExpectedValue_hyp -ExpectedValue_opt

            #Save Results
            Price_Chosen.append(action[0])
            Price_Optimal.append(MainPrice)
            Investment_Chosen.append(action[1])
            Investment_Optimal.append(MainInvestment)
            Deviation_Costs.append(DeviationCost)

    Price_Deviation = (np.array(Price_Chosen) - np.array(Price_Optimal))

    Investment_Deviation = (np.array(Investment_Chosen) - np.array(Investment_Optimal))

    Deviation_Costs = np.array(Deviation_Costs)

    Leader_Nash_Price = config.LeaderP
    Follower_Nash_Price = config.FollowerP
    Leader_Nash_Investment = config.LeaderX
    Follower_Nash_Investment = config.FollowerX


    # =========================================================
    # COLOUR PRICE POINTS BASED ON CHOSEN PRICE
    # =========================================================

    price_colours = np.where(
        np.array(Price_Chosen) > Leader_Nash_Price,
        'red',
        np.where(
            np.array(Price_Chosen) >= Follower_Nash_Price,
            'blue',
            'green'
        )
    )


    # =========================================================
    # COLOUR INVESTMENT POINTS
    # =========================================================
    # Replace these with your investment benchmarks if you have them

    investment_colours = np.where(
        np.array(Investment_Chosen) < Leader_Nash_Investment,
        'red',
        np.where(
            np.array(Investment_Chosen) <= Follower_Nash_Investment,
            'blue',
            'green'
        )
    )


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


    # =========================================================
    # PRICE
    # =========================================================

    ax1.scatter(
        Price_Deviation,
        Deviation_Costs,
        c=price_colours,
        s=8,
        alpha=0.25
    )

    # Zero deviation
    ax1.axvline(
        0,
        color='orange',
        linestyle='--',
        linewidth=2
    )

    # ----- Trend before zero -----

    mask_before = Price_Deviation < 0

    coef_before = np.polyfit(
        Price_Deviation[mask_before],
        Deviation_Costs[mask_before],
        1
    )

    x_before = np.linspace(
        Price_Deviation[mask_before].min(),
        0,
        100
    )

    ax1.plot(
        x_before,
        np.poly1d(coef_before)(x_before),
        color='black',
        linewidth=2
    )


    # ----- Trend after zero -----

    mask_after = Price_Deviation > 0

    coef_after = np.polyfit(
        Price_Deviation[mask_after],
        Deviation_Costs[mask_after],
        1
    )

    x_after = np.linspace(
        0,
        Price_Deviation[mask_after].max(),
        100
    )

    ax1.plot(
        x_after,
        np.poly1d(coef_after)(x_after),
        color='black',
        linewidth=2
    )

    ax1.set_xlabel('Price Deviation (Chosen − Optimal)')
    ax1.set_ylabel('Deviation Cost')
    ax1.set_title('Reward/Punishment Following Price Deviations')
    price_legend = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Above Leader Price',
            markerfacecolor='red',
            markersize=7
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Between Leader and Follower Prices',
            markerfacecolor='blue',
            markersize=7
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Below Follower Price',
            markerfacecolor='green',
            markersize=7
        )
    ]

    ax1.legend(
        handles=price_legend,
        loc='best'
    )


    # =========================================================
    # INVESTMENT
    # =========================================================

    ax2.scatter(
        Investment_Deviation,
        Deviation_Costs,
        c=investment_colours,
        s=8,
        alpha=0.25
    )

    # Zero deviation
    ax2.axvline(
        0,
        color='orange',
        linestyle='--',
        linewidth=2
    )

    # ----- Trend before zero -----

    mask_before = Investment_Deviation < 0

    coef_before = np.polyfit(
        Investment_Deviation[mask_before],
        Deviation_Costs[mask_before],
        1
    )

    x_before = np.linspace(
        Investment_Deviation[mask_before].min(),
        0,
        100
    )

    ax2.plot(
        x_before,
        np.poly1d(coef_before)(x_before),
        color='black',
        linewidth=2
    )


    # ----- Trend after zero -----

    mask_after = Investment_Deviation > 0

    coef_after = np.polyfit(
        Investment_Deviation[mask_after],
        Deviation_Costs[mask_after],
        1
    )

    x_after = np.linspace(
        0,
        Investment_Deviation[mask_after].max(),
        100
    )

    ax2.plot(
        x_after,
        np.poly1d(coef_after)(x_after),
        color='black',
        linewidth=2
    )

    ax2.set_xlabel('Investment Deviation (Chosen − Optimal)')
    ax2.set_ylabel('Deviation Cost')
    ax2.set_title('Reward/Punishment Following Investment Deviations')

    investment_legend = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Below Leader Investment',
            markerfacecolor='red',
            markersize=7
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Between Leader and Follower Investments',
            markerfacecolor='blue',
            markersize=7
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Above Follower Investment',
            markerfacecolor='green',
            markersize=7
        )
    ]

    ax2.legend(
        handles=investment_legend,
        loc='best'
    )


    plt.tight_layout()


    plt.savefig(r"TrainingResults\Reactions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def Welfare_Plot(CS_Theory,CS_Real,M_Theory,M_Real,save_path="TrainingResults/Welfare.png", show=False, dpi=300):
    print("Welfare")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 8))

    #helper smothing function
    def smooth_data(data, window):
        if len(data) < window:
            return data  # Return original if data is too short
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    CS_Real = np.array(CS_Real)
    CS_Theory = np.array(CS_Theory)
    windowsz = 100

    CS_Real_smooth = smooth_data(CS_Real, windowsz)
    CS_Theory_smooth = smooth_data(CS_Theory,windowsz)
    
    # --- Top Graph: CS Plot ---
    ax1.plot(CS_Theory_smooth, label="CS Theory", linestyle="--", color="blue")
    ax1.plot(CS_Real_smooth, label="CS Real", linestyle="-", color="darkblue")
    ax1.set_title("Welfare Comparison: Theory vs Real")
    ax1.set_ylabel("CS Values")
    ax1.legend()
    ax1.grid(True) # Adds a clean background grid
    
    # --- Bottom Graph: M Plot ---
    ax2.plot(M_Theory, label="M Theory", linestyle="--", color="orange")
    ax2.plot(M_Real, label="M Real", linestyle="-", color="darkorange")
    ax2.set_title("M Comparison: Theory vs Real")
    ax2.set_xlabel("Round") # X-axis label only goes on the bottom graph
    ax2.set_ylabel("M Values")
    ax2.legend()
    ax2.grid(True)
    
    # 2. Automatically clean up layout spacing so text doesn't overlap
    plt.tight_layout()
    
    # 3. Display the single window containing both plots

    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
  
#Save Distinct parquet file to training results with aggregated results
def Routine_Results(PriceStat,InvestStat,ProfitStat,State_logs,CS_Theory,CS_Real,M_Theory,M_Real,MarketShares,TotalRounds,config,TestParameter,ParIt):
    OutputLoc = "TrainingResultsC"
    data = {}

    #___Benchmarks___
    MonopolyB_Price = config.MonopolyP
    FollowerB_Price = config.FollowerP
    LeaderB_Price = config.LeaderP

    MonopolyB_Invest = config.MonopolyX
    FollowerB_Invest = config.FollowerX
    LeaderB_Invest = config.LeaderX

    MonopolyLB_Profit = config.MonopolyLeaderProfit
    MonopolyFB_Profit = config.MonopolyFollowerProfit
    FollowerB_Profit = config.FollowerProfit
    LeaderB_Profit = config.LeaderProfit

    MonopolyLB_MrktShr = config.MonopolyLeaderMrktShr
    MonopolyFB_MrktShr =config.MonopolyFollowerMrktShr
    LeaderB_MrktShr = config.LeaderMrktShr
    FollowerB_MrktShr = config.FollowerMrktShr
    
    #_Avg values_ (stat log)
    price_matrix = np.array(PriceStat)
    invest_matrix = np.array(InvestStat)
    profit_matrix = np.array(ProfitStat)
    mrktshr_matrix = np.array(MarketShares)

    firm_price_avgs = price_matrix[:, :-1].mean(axis=0)
    firm_invest_avgs = invest_matrix[:, :-1].mean(axis=0)
    firm_profit_avgs = profit_matrix[:, :-1].mean(axis=0)
    firm_mrktshr_avgs = mrktshr_matrix[:, :-1].mean(axis=0)

    #Save results
    for idx in range(len(firm_profit_avgs)):

        data[f'Price_F{idx}'] = firm_price_avgs[idx]
        data[f'Invest_F{idx}'] = firm_invest_avgs[idx]
        data[f'Profit_F{idx}'] = firm_profit_avgs[idx]
        data[f'MrktShr_F{idx}'] = firm_mrktshr_avgs[idx]

        if config.firms>1:
            #Prices
            data['LeaderPrice_Benchmark'] = LeaderB_Price
            data['FollowerPrice_Benchmark'] = FollowerB_Price
            #Investments
            data['LeaderInvest_Benchmark'] = LeaderB_Invest
            data['FollowerInvest_Benchmark'] = FollowerB_Invest
            #Profit
            data['LeaderProfit_Benchmark'] = LeaderB_Profit
            data['FollowerProfit_Benchmark'] = FollowerB_Profit
            data['MonopolyFProfit_Benchmark'] = MonopolyFB_Profit
            #Marketshare
            data['LeaderMrktShr_Benchmark'] = LeaderB_MrktShr
            data['FollowerMrktShr_Benchmark'] = FollowerB_MrktShr
            data['MonopolyFMrktShr_Benchmark'] = MonopolyFB_MrktShr

        data['MonopolyPrice_Benchmark'] = MonopolyB_Price
        data['MonopolyInvest_Benchmark'] = MonopolyB_Invest
        data['MonopolyLProfit_Benchmark'] = MonopolyLB_Profit
        data['MonopolyLMrktShr_Benchmark'] = MonopolyLB_MrktShr
        

    #_Avg when firm x is leader_
        #_Leader Indexes
    Leader_Indexes = price_matrix[:, -1]
    cleaned_indexes = [x for x in Leader_Indexes if x is not None]
    MrktLeaderIndxs = np.unique(cleaned_indexes)

    for leader in MrktLeaderIndxs:
        mask = Leader_Indexes == leader #mask will be the same for all matrices
        leader_avg_price = price_matrix[mask][:, :-1].mean(axis=0)
        leader_avg_invest = invest_matrix[mask][:, :-1].mean(axis=0)
        leader_avg_profit = profit_matrix[mask][:, :-1].mean(axis=0)
        leader_avg_mrktshr = mrktshr_matrix[mask][:, :-1].mean(axis=0)

        #Save results
        for idx in range(len(leader_avg_price)):
           
            #Prices
            data[f'Price_F{idx}_Leader{int(leader)}'] = leader_avg_price[idx]
            #Investments
            data[f'Invest_F{idx}_Leader{int(leader)}'] = leader_avg_invest[idx]
            #Profit
            data[f'Profit_F{idx}_Leader{int(leader)}'] = leader_avg_profit[idx]
            #Marketshare
            data[f'MrktShr_F{idx}_Leader{int(leader)}'] = leader_avg_mrktshr[idx]

            
    #_Welfare_
    CSTheory_Avg = np.mean(CS_Theory)
    CSReal_Avg = np.mean(CS_Real)
    CSReal_V_CSTheory = (CSReal_Avg - CSTheory_Avg)/CSTheory_Avg *100

    MT = pd.Series(M_Theory)
    MR = pd.Series(M_Real)

    MT_avg_pct_chng = MT.pct_change().mean()
    MR_avg_pct_chng = MR.pct_change().mean() 

    MRPct_V_MTPct = (MR_avg_pct_chng - MT_avg_pct_chng)/MT_avg_pct_chng *100 if MT_avg_pct_chng !=0 else 0

    data[f'ConsumerSurplus'] = CSReal_V_CSTheory
    data[f'IndustryMPct'] = MRPct_V_MTPct

    #_Strategy_
    Explength = config.explorationlen
    ConvergenceTime = TotalRounds - Explength
    UniqueStates = len(State_logs)

    data[f'ConvergTime'] = ConvergenceTime
    data[f'UniqueStates'] = UniqueStates

    df = pd.DataFrame(data,index=[0])
    
    # Save as binary parquet 
    if not TestParameter: #no parameters passed
        key, value = "test", ""
    else:
        key, value = next(iter(TestParameter.items()))

    df.to_parquet(f"{OutputLoc}/run_{key}_{value}_{ParIt}.parquet")

def PlotParallel(value_name,marketleader=None):
    parquet_folder="TrainingResultsC"
    title = value_name
    
    if marketleader is not None:
        save_path = f"TrainingResults/parallel_{value_name.lower()}_firm{marketleader}_boxplots.png"
    else:
        save_path = f"TrainingResults/parallel_{value_name.lower()}_boxplots.png"

    ParquetFolder = Path(parquet_folder)
    def convert_parameter(value):
        try:
            num = float(value)

            # Keep integers as integers
            if num.is_integer():
                return int(num)

            return num

        except ValueError:
            return value

    #Group files by tested parameter
    grouped_files = defaultdict(list)
    for file in ParquetFolder.glob("*.parquet"):
        file_name = file.name

        #Safety to make sure file matches naming convention
        if file_name.startswith("run_") and "_" in file_name:
            details = file_name.split("_")

            tested_parameter = details[1] #get parameter name
            parameter_value = convert_parameter(details[2])
            df = pd.read_parquet(file) #load parquet information into dataframe obj

            grouped_files[parameter_value].append(df)

    final_data_dict = {}
    for param, df_list in grouped_files.items():
        final_data_dict[param] = pd.concat(df_list, ignore_index=True)
    plot_rows = []
    for param_val,curr_df in final_data_dict.items():
        if value_name == "Price":
            if marketleader is not None:
                matched_columns = curr_df.filter(regex=fr'^Price_F\d+_Leader{int(marketleader)}$')
                
                if not curr_df.filter(regex=fr'^LeaderPrice_Benchmark').empty:
                    leader_benchmark = curr_df.filter(regex=fr'^LeaderPrice_Benchmark').squeeze().unique().tolist()
                    follower_benchmark = curr_df.filter(regex=fr'^FollowerPrice_Benchmark').squeeze().unique().tolist()
                monopolyL_benchmark = curr_df.filter(regex=fr'^MonopolyPrice_Benchmark').squeeze().unique().tolist()
            else:   
                matched_columns = curr_df.filter(regex=r'^Price_F\d+$')
                if not curr_df.filter(regex=fr'^LeaderPrice_Benchmark').empty:
                    leader_benchmark = curr_df.filter(regex=r'^LeaderPrice_Benchmark').squeeze().unique().tolist()
                    follower_benchmark = curr_df.filter(regex=r'^FollowerPrice_Benchmark').squeeze().unique().tolist()
                monopolyL_benchmark = curr_df.filter(regex=r'^MonopolyPrice_Benchmark').squeeze().unique().tolist()
            monopolyF_benchmark = [None]
            if curr_df.filter(regex=fr'^LeaderPrice_Benchmark').empty:
                leader_benchmark = [None]
                follower_benchmark = [None]

        elif value_name =="Invest":
            if marketleader is not None:
                matched_columns = curr_df.filter(regex=fr'^Invest_F\d+_Leader{int(marketleader)}$')
                if not curr_df.filter(regex=fr'^LeaderInvest_Benchmark').empty:
                    leader_benchmark = curr_df.filter(regex=fr'^LeaderInvest_Benchmark').squeeze().unique().tolist()
                    follower_benchmark = curr_df.filter(regex=fr'^FollowerInvest_Benchmark').squeeze().unique().tolist()
                monopolyL_benchmark = curr_df.filter(regex=fr'^MonopolyInvest_Benchmark').squeeze().unique().tolist()
            else:
                matched_columns = curr_df.filter(regex=r'^Invest_F\d+$')
                if not curr_df.filter(regex=fr'^LeaderInvest_Benchmark').empty:
                    leader_benchmark = curr_df.filter(regex=r'^LeaderInvest_Benchmark').squeeze().unique().tolist()
                    follower_benchmark = curr_df.filter(regex=r'^FollowerInvest_Benchmark').squeeze().unique().tolist()
                monopolyL_benchmark = curr_df.filter(regex=r'^MonopolyInvest_Benchmark').squeeze().unique().tolist()
            monopolyF_benchmark = [None]
            if curr_df.filter(regex=fr'^LeaderInvest_Benchmark').empty:
                leader_benchmark = [None]
                follower_benchmark = [None]

        elif value_name =="Profit":
            if marketleader is not None:
                matched_columns = curr_df.filter(regex=fr'^Profit_F\d+_Leader{int(marketleader)}$')
                if not curr_df.filter(regex=fr'^LeaderProfit_Benchmark').empty:
                    leader_benchmark = curr_df.filter(regex=fr'^LeaderProfit_Benchmark').squeeze().unique().tolist()
                    follower_benchmark = curr_df.filter(regex=fr'^FollowerProfit_Benchmark').squeeze().unique().tolist()
                    monopolyF_benchmark = curr_df.filter(regex=fr'^MonopolyFProfit_Benchmark').squeeze().unique().tolist()
                monopolyL_benchmark = curr_df.filter(regex=fr'^MonopolyLProfit_Benchmark').squeeze().unique().tolist()
                
            else:
                matched_columns = curr_df.filter(regex=r'^Profit_F\d+$')
                if not curr_df.filter(regex=fr'^LeaderProfit_Benchmark').empty:
                    leader_benchmark = curr_df.filter(regex=r'^LeaderProfit_Benchmark').squeeze().unique().tolist()
                    follower_benchmark = curr_df.filter(regex=r'^FollowerProfit_Benchmark').squeeze().unique().tolist()
                    monopolyF_benchmark = curr_df.filter(regex=r'^MonopolyFProfit_Benchmark').squeeze().unique().tolist()
                monopolyL_benchmark = curr_df.filter(regex=r'^MonopolyLProfit_Benchmark').squeeze().unique().tolist()
            if curr_df.filter(regex=fr'^LeaderProfit_Benchmark').empty:
                leader_benchmark = [None]
                follower_benchmark = [None]
                monopolyF_benchmark = [None]
                
        elif value_name =="MrktShr":
            if marketleader is not None:
                matched_columns = curr_df.filter(regex=fr'^MrktShr_F\d+_Leader{int(marketleader)}$')
                if not curr_df.filter(regex=fr'^LeaderMrktShr_Benchmark').empty:
                    leader_benchmark = curr_df.filter(regex=fr'^LeaderMrktShr_Benchmark').squeeze().unique().tolist()
                    follower_benchmark = curr_df.filter(regex=fr'^FollowerMrktShr_Benchmark').squeeze().unique().tolist()
                    monopolyF_benchmark = curr_df.filter(regex=fr'^MonopolyFMrktShr_Benchmark').squeeze().unique().tolist()
                monopolyL_benchmark = curr_df.filter(regex=fr'^MonopolyLMrktShr_Benchmark').squeeze().unique().tolist()
                
            else:
                matched_columns = curr_df.filter(regex=r'^MrktShr_F\d+$')
                if not curr_df.filter(regex=fr'^LeaderMrktShr_Benchmark').empty:
                    leader_benchmark = curr_df.filter(regex=r'^LeaderMrktShr_Benchmark').squeeze().unique().tolist()
                    follower_benchmark = curr_df.filter(regex=r'^FollowerMrktShr_Benchmark').squeeze().unique().tolist()
                    monopolyF_benchmark = curr_df.filter(regex=r'^MonopolyFMrktShr_Benchmark').squeeze().unique().tolist()
                monopolyL_benchmark = curr_df.filter(regex=r'^MonopolyLMrktShr_Benchmark').squeeze().unique().tolist()
            if curr_df.filter(regex=fr'^LeaderMrktShr_Benchmark').empty:
                leader_benchmark = [None]
                follower_benchmark = [None]
                monopolyF_benchmark = [None]
                

        #should only be 1 unique benchmark per column
        if len(leader_benchmark)>1 or len(follower_benchmark)>1 or len(monopolyL_benchmark)>1 or len(monopolyF_benchmark)>1:
            raise ValueError("More than 1 unique benchmark")
        
        for idx, firm_column in enumerate(matched_columns.columns, start=1):
    
            
            for row_index, value in matched_columns[firm_column].items():
                plot_rows.append({
                            "Parameter": param_val,
                            "Firm": f"Firm {idx}",
                            value_name:value,
                            "Leader_BM":leader_benchmark[0],
                            "Follower_BM":follower_benchmark[0],
                            "MonopolyL_BM":monopolyL_benchmark[0],
                            "MonopolyF_BM":monopolyF_benchmark[0]

                        })
            
    combined_values = pd.DataFrame(plot_rows)
    combined_values = combined_values.sort_values("Parameter")

    plt.figure(figsize=(10, 6))

    # Create the box plot grouped by Parameter and split by Firm
    sns.boxplot(data=combined_values, x="Parameter", y=value_name, hue="Firm",palette="Set2")

    num_categories = combined_values["Parameter"].nunique()
    dynamic_dash_size = max(20, 300 / num_categories)

    #Add benchmarks
    benchmark_cols = ["Leader_BM", "Follower_BM", "MonopolyL_BM", "MonopolyF_BM"]
    colors = ["#1565C0", "#2E7D32", "#C62828", "#8E24AA"]
    benchmark_palette = dict(zip(benchmark_cols, colors))

    # 2. Melt only the columns we need so Seaborn can handle them all at once
    benchmarks_melted = combined_values.melt(
        id_vars=["Parameter"], 
        value_vars=benchmark_cols, 
        var_name="Benchmark", 
        value_name="Benchmark_Value"
    )

    # 3. Plot ALL benchmarks in a single call so dodge=True can place them side-by-side
    sns.stripplot(
        data=benchmarks_melted,
        x="Parameter",
        y="Benchmark_Value",
        hue="Benchmark",          # Group by the benchmark identity
        palette=benchmark_palette,# Pass the specific colors mapping safely
        dodge=True,               # Cleanly splits the width into half/fourths
        marker="_",
        size=dynamic_dash_size, 
        linewidth=1.5,            # Increased slightly for better visibility over boxes
        jitter=False,        
        legend=True               # Keep true so your legend dictionary catches them!
    )

    #legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = plt.legend(
        by_label.values(), 
        by_label.keys(), 
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=3,                     
        frameon=True,              
        borderaxespad=0.5
    )
   
    for handle in leg.legend_handles:
        if hasattr(handle, 'set_markersize'):
            handle.set_markersize(20)
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([40])
        if hasattr(handle, 'set_linewidth'):
            handle.set_linewidth(0.5) 
    
    # Clean up the labels and title
    plt.title(title)
    plt.xlabel(f"Parameter - {tested_parameter}")
    plt.ylabel(value_name)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Display the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()
   
def PlotWelfareParallel():
    parquet_folder="TrainingResultsC"
    save_path_CS = f"TrainingResults/parallel_CS_boxplots.png"
    save_path_M = f"TrainingResults/parallel_Innovation_boxplots.png"

    ParquetFolder = Path(parquet_folder)
    def convert_parameter(value):
        try:
            num = float(value)

            # Keep integers as integers
            if num.is_integer():
                return int(num)

            return num

        except ValueError:
            return value

    #Group files by tested parameter
    grouped_files = defaultdict(list)
    for file in ParquetFolder.glob("*.parquet"):
        file_name = file.name

        #Safety to make sure file matches naming convention
        if file_name.startswith("run_") and "_" in file_name:
            details = file_name.split("_")

            tested_parameter = details[1] #get parameter name
            parameter_value = convert_parameter(details[2])

            df = pd.read_parquet(file) #load parquet information into dataframe obj

            grouped_files[parameter_value].append(df)

    final_data_dict = {}
    for param, df_list in grouped_files.items():
        final_data_dict[param] = pd.concat(df_list, ignore_index=True)

    CS_plot_rows = []
    M_Val_rows = []
    for key,curr_df in final_data_dict.items():
        matched_columns_CS = curr_df.filter(regex='ConsumerSurplus')
        matched_columns_M = curr_df.filter(regex='IndustryMPct')

        flat_values_CS = matched_columns_CS.to_numpy().flatten()
        flat_values_M = matched_columns_M.to_numpy().flatten()

        for value in flat_values_CS:
            CS_plot_rows.append({'Key': key,'Consumer Surplus': value})

        for value in flat_values_M:
            M_Val_rows.append({
                'Key': key,
                'Innovation': value
            })

    df_melted_CS = pd.DataFrame(CS_plot_rows)
    df_melted_M = pd.DataFrame(M_Val_rows)

    df_melted_CS = df_melted_CS.sort_values("Key")
    df_melted_M = df_melted_M.sort_values("Key")

    # Create the Box Plot - consumer surplus
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Key", 
        y="Consumer Surplus", 
        data=df_melted_CS, 
        palette="Set2",
        hue="Key",      
        legend=False        
    )
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    plt.text(
            x=-0.45,
            y=0,
            s="Benchmark",
            color="black",
            va="bottom",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
            zorder=3,
        )

    

    plt.title(f"Consumer Surplus Distribution", fontsize=14)
    plt.xlabel(f"Tested Parameter - {tested_parameter}", fontsize=12)
    plt.ylabel("Consumer Surplus Difference (%)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path_CS, dpi=300, bbox_inches='tight')
    plt.close()

    # Create the Box Plot - Welfare
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Key", 
        y="Innovation", 
        data=df_melted_M, 
        palette="Set2", 
        hue="Key",      
        legend=False        
    )
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    plt.text(
            x=-0.45,
            y=0,
            s="Benchmark",
            color="black",
            va="bottom",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
            zorder=3,
        )



    plt.title(f"Innovation Rate Distribution", fontsize=14)
    plt.xlabel(f"Tested Parameter - {tested_parameter}", fontsize=12)
    plt.ylabel("Benchmark Innovation Rate vs Market (%)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path_M, dpi=300, bbox_inches='tight')
    plt.close()

def PlotStrategyParallel():
    parquet_folder="TrainingResultsC"
    save_path_CS = f"TrainingResults/parallel_convergence_boxplots.png"
    save_path_M = f"TrainingResults/parallel_states_boxplots.png"

    ParquetFolder = Path(parquet_folder)

    def convert_parameter(value):
        try:
            num = float(value)

            # Keep integers as integers
            if num.is_integer():
                return int(num)

            return num

        except ValueError:
            return value

    #Group files by tested parameter
    grouped_files = defaultdict(list)
    for file in ParquetFolder.glob("*.parquet"):
        file_name = file.name

        #Safety to make sure file matches naming convention
        if file_name.startswith("run_") and "_" in file_name:
            details = file_name.split("_")

            tested_parameter = details[1] #get parameter name 
            parameter_value = convert_parameter(details[2])
            df = pd.read_parquet(file) #load parquet information into dataframe obj

            grouped_files[parameter_value].append(df)

    final_data_dict = {}
    for param, df_list in grouped_files.items():
        final_data_dict[param] = pd.concat(df_list, ignore_index=True)

    strategy_plot_rows = []
    states_plot_rows = []
    for key,curr_df in final_data_dict.items():
        matched_columns_strategy = curr_df.filter(regex='ConvergTime')
        matched_columns_states= curr_df.filter(regex='UniqueStates')

        flat_values_strategy = matched_columns_strategy.to_numpy().flatten()
        flat_values_states = matched_columns_states.to_numpy().flatten()

        for value in flat_values_strategy:
            strategy_plot_rows.append({'Key': key,'Convergence Time': value})

        for value in flat_values_states:
            states_plot_rows.append({
                'Key': key,
                'Unique States': value
            })

    df_melted_strategy = pd.DataFrame(strategy_plot_rows)
    df_melted_states = pd.DataFrame(states_plot_rows)
    df_melted_strategy = df_melted_strategy.sort_values("Key")
    df_melted_states = df_melted_states.sort_values("Key")

    # Create the Box Plot - consumer surplus
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Key", 
        y="Convergence Time", 
        data=df_melted_strategy, 
        palette="Set2", 
        hue="Key",      
        legend=False        
    )

    plt.title(f"Market Convergence Time", fontsize=14)
    plt.xlabel(f"Tested Parameter - {tested_parameter}", fontsize=12)
    plt.ylabel("Iterations", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path_CS, dpi=300, bbox_inches='tight')
    plt.close()

    # Create the Box Plot - Welfare
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Key", 
        y="Unique States", 
        data=df_melted_states, 
        palette="Set2",
        hue="Key",      
        legend=False        
    )

    plt.title(f"Number of Unique States", fontsize=14)
    plt.xlabel(f"Tested Parameter - {tested_parameter}", fontsize=12)
    plt.ylabel("Unique States", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path_M, dpi=300, bbox_inches='tight')
    plt.close()

def Parallel_Outputs(MaxFirms):

    #Normal plot
    PlotParallel(value_name = "Price")
    PlotParallel(value_name = "Invest")
    PlotParallel(value_name = "Profit")
    PlotParallel(value_name = "MrktShr")
        
    #Leader Plot
    for leader in range(MaxFirms):
        PlotParallel(value_name = "Price",marketleader=leader)
        PlotParallel(value_name = "Invest",marketleader=leader)
        PlotParallel(value_name = "Profit",marketleader=leader)
        PlotParallel(value_name = "MrktShr",marketleader=leader)

    #Welfare Plot
    PlotWelfareParallel()

    #Strategy Plot
    PlotStrategyParallel()