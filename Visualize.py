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
            leaderI_turns_stat = matrix_s[leader_mask_stat]

            leaderI_exp_points = len(leaderI_turns_exp)
            leaderI_stat_points = len(leaderI_turns_stat)

            exp_turns = np.arange(leaderI_exp_points) * downsample
            stat_turns = np.arange(leaderI_stat_points) + max(exp_turns) +1

            total_turns = np.concatenate([exp_turns, stat_turns])
            total_points = np.concatenate([leaderI_turns_exp,leaderI_turns_stat])

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
                #plot last 100 points to see if there is oscillation
                ax.scatter(
                    stat_turns[-100:],
                    leaderI_turns_stat[:, firm_index][-100:], #showing last 100
                    color=firm_color,
                    s=10,
                    zorder=3
                )

            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} - Firm {leader_index + 1} Leader")
            add_equilibrium_lines(ax, title, config, label_x)
            ax.grid(True, alpha=0.25)
            ax.margins(x=0)

        #Hide y axis on all column except first one (save space)
        if leader_index > 0:
            for row_index in range(3):
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
def Routine_Results(PriceStat,InvestStat,ProfitStat,State_logs,CS_Theory,CS_Real,M_Theory,M_Real,TotalRounds,config,TestParameter,ParIt):
    OutputLoc = "TrainingResultsC"
    data = {"param_id":1, "run_id":1}

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

    #_Avg values_ (stat log)
    price_matrix = np.array(PriceStat)
    invest_matrix = np.array(InvestStat)
    profit_matrix = np.array(ProfitStat)

    firm_price_avgs = price_matrix[:, :-1].mean(axis=0)
    firm_invest_avgs = invest_matrix[:, :-1].mean(axis=0)
    firm_profit_avgs = profit_matrix[:, :-1].mean(axis=0)

        #Compare to benchmarks
    Price_V_Monopoly = firm_price_avgs - MonopolyB_Price
    Price_V_Follower = firm_price_avgs - FollowerB_Price
    Price_V_Leader = firm_price_avgs - LeaderB_Price

    Invest_V_Monopoly = firm_invest_avgs - MonopolyB_Invest
    Invest_V_Follower = firm_invest_avgs - FollowerB_Invest
    Invest_V_Leader = firm_invest_avgs - LeaderB_Invest

    Profit_V_MonopolyL = firm_profit_avgs - MonopolyLB_Profit
    Profit_V_MonopolyF = firm_profit_avgs - MonopolyFB_Profit
    Profit_V_Follower = firm_profit_avgs - FollowerB_Profit
    Profit_V_Leader = firm_profit_avgs - LeaderB_Profit

    #Save results
    for idx in range(len(Price_V_Monopoly)):
        data[f'MonopolyPrice_F{idx}'] = Price_V_Monopoly[idx]
        data[f'LeaderPrice_F{idx}'] = Price_V_Leader[idx]
        data[f'FollowerPrice_F{idx}'] =Price_V_Follower[idx]

        data[f'MonopolyInvest_F{idx}'] = Invest_V_Monopoly[idx]
        data[f'LeaderInvest_F{idx}'] = Invest_V_Leader[idx]
        data[f'FollowerInvest_F{idx}'] =Invest_V_Follower[idx]

        data[f'MonopolyLProfit_F{idx}'] = Profit_V_MonopolyL[idx]
        data[f'MonopolyFProfit_F{idx}'] = Profit_V_MonopolyF[idx]
        data[f'LeaderProfit_F{idx}'] = Profit_V_Leader[idx]
        data[f'FollowerProfit_F{idx}'] = Profit_V_Follower[idx]

    #_Avg when firm x is leader_
        #_Leader Indexes
    Leader_Indexes = price_matrix[:, -1]
    MrktLeaderIndxs = np.unique(Leader_Indexes)

    for leader in MrktLeaderIndxs:
        mask = Leader_Indexes == leader #mask will be the same for all matrices
        leader_avg_price = price_matrix[mask][:, :-1].mean(axis=0)
        leader_avg_invest = invest_matrix[mask][:, :-1].mean(axis=0)
        leader_avg_profit = profit_matrix[mask][:, :-1].mean(axis=0)

        #Compare to benchmarks
        LPrice_V_Monopoly = leader_avg_price - MonopolyB_Price
        LPrice_V_Follower = leader_avg_price - FollowerB_Price
        LPrice_V_Leader = leader_avg_price - LeaderB_Price

        LInvest_V_Monopoly = leader_avg_invest - MonopolyB_Invest
        LInvest_V_Follower = leader_avg_invest - FollowerB_Invest
        LInvest_V_Leader = leader_avg_invest - LeaderB_Invest

        LProfit_V_MonopolyL = leader_avg_profit - MonopolyLB_Profit
        LProfit_V_MonopolyF = leader_avg_profit - MonopolyFB_Profit
        LProfit_V_Follower = leader_avg_profit - FollowerB_Profit
        LProfit_V_Leader = leader_avg_profit - LeaderB_Profit

        #Save results

        for idx in range(len(LPrice_V_Monopoly)):
            data[f'Leader{leader}_MonopolyPrice_F{idx}'] = LPrice_V_Monopoly[idx]
            data[f'Leader{leader}_LeaderPrice_F{idx}'] = LPrice_V_Leader[idx]
            data[f'Leader{leader}_FollowerPrice_F{idx}'] =LPrice_V_Follower[idx]

            data[f'Leader{leader}_MonopolyInvest_F{idx}'] = LInvest_V_Monopoly[idx]
            data[f'Leader{leader}_LeaderInvest_F{idx}'] = LInvest_V_Leader[idx]
            data[f'Leader{leader}_FollowerInvest_F{idx}'] =LInvest_V_Follower[idx]

            data[f'Leader{leader}_MonopolyLProfit_F{idx}'] = LProfit_V_MonopolyL[idx]
            data[f'Leader{leader}_MonopolyFProfit_F{idx}'] = LProfit_V_MonopolyF[idx]
            data[f'Leader{leader}_LeaderProfit_F{idx}'] = LProfit_V_Leader[idx]
            data[f'Leader{leader}_FollowerProfit_F{idx}'] = LProfit_V_Follower[idx]

    #_Welfare_
    CSTheory_Avg = np.mean(CS_Theory)
    CSReal_Avg = np.mean(CS_Real)
    CSReal_V_CSTheory = CSReal_Avg - CSTheory_Avg

    MT = pd.Series(M_Theory)
    MR = pd.Series(M_Real)

    MT_avg_pct_chng = MT.pct_change().mean() * 100
    MR_avg_pct_chng = MR.pct_change().mean() * 100

    MRPct_V_MTPct = MR_avg_pct_chng - MT_avg_pct_chng

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

    df.to_parquet(f"{OutputLoc}/run_{key}{value}_{ParIt}.parquet")

