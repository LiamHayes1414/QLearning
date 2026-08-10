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
import re

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
    firm_mrktshr_avg = mrktshr_matrix[:, :-1].mean(axis=0)

    def benchmark_comparison(values, benchmark,Percent=True):
        if Percent:
            return (values - benchmark)/benchmark *100, "Percent"
        return values - benchmark, "Difference"
        
    def save_result(column, value, unit):
        data[column] = value
        data[f"{column}_Unit"] = unit

    #Compare to benchmarks
    if config.firms>1:
        Price_V_Follower, Price_V_Follower_Unit = benchmark_comparison(firm_price_avgs, FollowerB_Price)
        Price_V_Leader, Price_V_Leader_Unit = benchmark_comparison(firm_price_avgs, LeaderB_Price)

        Invest_V_Follower, Invest_V_Follower_Unit = benchmark_comparison(firm_invest_avgs, FollowerB_Invest,False)
        Invest_V_Leader, Invest_V_Leader_Unit = benchmark_comparison(firm_invest_avgs, LeaderB_Invest,False)

        Profit_V_MonopolyF, Profit_V_MonopolyF_Unit = benchmark_comparison(firm_profit_avgs, MonopolyFB_Profit)
        Profit_V_Follower, Profit_V_Follower_Unit = benchmark_comparison(firm_profit_avgs, FollowerB_Profit)
        Profit_V_Leader, Profit_V_Leader_Unit = benchmark_comparison(firm_profit_avgs, LeaderB_Profit)

        MrktShr_V_MonopolyF,MrktShr_V_MonopolyF_Unit = benchmark_comparison(firm_mrktshr_avg, MonopolyFB_MrktShr,False)
        MrktShr_V_Leader,MrktShr_V_Leader_Unit = benchmark_comparison(firm_mrktshr_avg, LeaderB_MrktShr,False)
        MrktShr_V_Follower,MrktShr_V_Follower_Unit = benchmark_comparison(firm_mrktshr_avg, FollowerB_MrktShr,False)

    Price_V_Monopoly, Price_V_Monopoly_Unit = benchmark_comparison(firm_price_avgs, MonopolyB_Price)
    Invest_V_Monopoly, Invest_V_Monopoly_Unit = benchmark_comparison(firm_invest_avgs, MonopolyB_Invest,False)
    Profit_V_MonopolyL, Profit_V_MonopolyL_Unit = benchmark_comparison(firm_profit_avgs, MonopolyLB_Profit)
    MrktShr_V_MonopolyL,MrktShr_V_MonopolyL_Unit = benchmark_comparison(firm_mrktshr_avg, MonopolyLB_MrktShr,False)
    
    #Save results
    for idx in range(len(Price_V_Monopoly)):

        if config.firms>1:
            save_result(f'LeaderPrice_F{idx}', Price_V_Leader[idx], Price_V_Leader_Unit)
            save_result(f'FollowerPrice_F{idx}', Price_V_Follower[idx], Price_V_Follower_Unit)

            save_result(f'LeaderInvest_F{idx}', Invest_V_Leader[idx], Invest_V_Leader_Unit)
            save_result(f'FollowerInvest_F{idx}', Invest_V_Follower[idx], Invest_V_Follower_Unit)

            save_result(f'MonopolyFProfit_F{idx}', Profit_V_MonopolyF[idx], Profit_V_MonopolyF_Unit)
            save_result(f'LeaderProfit_F{idx}', Profit_V_Leader[idx], Profit_V_Leader_Unit)
            save_result(f'FollowerProfit_F{idx}', Profit_V_Follower[idx], Profit_V_Follower_Unit)

            save_result(f'MonopolyFMrktShr_F{idx}', MrktShr_V_MonopolyF[idx], MrktShr_V_MonopolyF_Unit)
            save_result(f'LeaderMrktShr_F{idx}', MrktShr_V_Leader[idx], MrktShr_V_Leader_Unit)
            save_result(f'FollowerMrktShr_F{idx}', MrktShr_V_Follower[idx], MrktShr_V_Follower_Unit)

        save_result(f'MonopolyPrice_F{idx}', Price_V_Monopoly[idx], Price_V_Monopoly_Unit)
        save_result(f'MonopolyInvest_F{idx}', Invest_V_Monopoly[idx], Invest_V_Monopoly_Unit)
        save_result(f'MonopolyLProfit_F{idx}', Profit_V_MonopolyL[idx], Profit_V_MonopolyL_Unit)
        save_result(f'MonopolyLMrktShr_F{idx}', MrktShr_V_MonopolyL[idx], MrktShr_V_MonopolyL_Unit)
        

    #_Avg when firm x is leader_
        #_Leader Indexes
    Leader_Indexes = price_matrix[:, -1]
    MrktLeaderIndxs = np.unique(Leader_Indexes)

    for leader in MrktLeaderIndxs:
        mask = Leader_Indexes == leader #mask will be the same for all matrices
        leader_avg_price = price_matrix[mask][:, :-1].mean(axis=0)
        leader_avg_invest = invest_matrix[mask][:, :-1].mean(axis=0)
        leader_avg_profit = profit_matrix[mask][:, :-1].mean(axis=0)
        leader_avg_mrktshr = mrktshr_matrix[mask][:, :-1].mean(axis=0)

        #Compare to benchmarks
        if config.firms>1:
            LPrice_V_Follower, LPrice_V_Follower_Unit = benchmark_comparison(leader_avg_price, FollowerB_Price)
            LPrice_V_Leader, LPrice_V_Leader_Unit = benchmark_comparison(leader_avg_price, LeaderB_Price)

            LInvest_V_Follower, LInvest_V_Follower_Unit = benchmark_comparison(leader_avg_invest, FollowerB_Invest,False)
            LInvest_V_Leader, LInvest_V_Leader_Unit = benchmark_comparison(leader_avg_invest, LeaderB_Invest,False)

            LProfit_V_MonopolyF, LProfit_V_MonopolyF_Unit = benchmark_comparison(leader_avg_profit, MonopolyFB_Profit)
            LProfit_V_Follower, LProfit_V_Follower_Unit = benchmark_comparison(leader_avg_profit, FollowerB_Profit)
            LProfit_V_Leader, LProfit_V_Leader_Unit = benchmark_comparison(leader_avg_profit, LeaderB_Profit)

            LMrktShr_V_MonopolyF,LMrktShr_V_MonopolyF_Unit = benchmark_comparison(leader_avg_mrktshr, MonopolyFB_MrktShr,False)
            LMrktShr_V_Leader,LMrktShr_V_Leader_Unit = benchmark_comparison(leader_avg_mrktshr, LeaderB_MrktShr,False)
            LMrktShr_V_Follower,LMrktShr_V_Follower_Unit = benchmark_comparison(leader_avg_mrktshr, FollowerB_MrktShr,False)

        LPrice_V_Monopoly, LPrice_V_Monopoly_Unit = benchmark_comparison(leader_avg_price, MonopolyB_Price)
        LInvest_V_Monopoly, LInvest_V_Monopoly_Unit = benchmark_comparison(leader_avg_invest, MonopolyB_Invest,False)
        LProfit_V_MonopolyL, LProfit_V_MonopolyL_Unit = benchmark_comparison(leader_avg_profit, MonopolyLB_Profit)
        LMrktShr_V_MonopolyL,LMrktShr_V_MonopolyL_Unit = benchmark_comparison(leader_avg_mrktshr, MonopolyLB_MrktShr,False)
    
        #Save results
        for idx in range(len(LPrice_V_Monopoly)):
            if config.firms>1:
                save_result(f'Leader{int(leader)}_LeaderPrice_F{idx}', LPrice_V_Leader[idx], LPrice_V_Leader_Unit)
                save_result(f'Leader{int(leader)}_FollowerPrice_F{idx}', LPrice_V_Follower[idx], LPrice_V_Follower_Unit)

                save_result(f'Leader{int(leader)}_LeaderInvest_F{idx}', LInvest_V_Leader[idx], LInvest_V_Leader_Unit)
                save_result(f'Leader{int(leader)}_FollowerInvest_F{idx}', LInvest_V_Follower[idx], LInvest_V_Follower_Unit)

                save_result(f'Leader{int(leader)}_MonopolyFProfit_F{idx}', LProfit_V_MonopolyF[idx], LProfit_V_MonopolyF_Unit)
                save_result(f'Leader{int(leader)}_LeaderProfit_F{idx}', LProfit_V_Leader[idx], LProfit_V_Leader_Unit)
                save_result(f'Leader{int(leader)}_FollowerProfit_F{idx}', LProfit_V_Follower[idx], LProfit_V_Follower_Unit)

                save_result(f'Leader{int(leader)}_MonopolyFMrktShr_F{idx}', LMrktShr_V_MonopolyF[idx], LMrktShr_V_MonopolyF_Unit)
                save_result(f'Leader{int(leader)}_LeaderMrktShr_F{idx}', LMrktShr_V_Leader[idx], LMrktShr_V_Leader_Unit)
                save_result(f'Leader{int(leader)}_FollowerMrktShr_F{idx}', LMrktShr_V_Follower[idx], LMrktShr_V_Follower_Unit)

            save_result(f'Leader{int(leader)}_MonopolyPrice_F{idx}', LPrice_V_Monopoly[idx], LPrice_V_Monopoly_Unit)
            save_result(f'Leader{int(leader)}_MonopolyInvest_F{idx}', LInvest_V_Monopoly[idx], LInvest_V_Monopoly_Unit)
            save_result(f'Leader{int(leader)}_MonopolyLProfit_F{idx}', LProfit_V_MonopolyL[idx], LProfit_V_MonopolyL_Unit)
            save_result(f'Leader{int(leader)}_MonopolyLMrktShr_F{idx}', LMrktShr_V_MonopolyL[idx], LMrktShr_V_MonopolyL_Unit)
            
    #_Welfare_
    CSTheory_Avg = np.mean(CS_Theory)
    CSReal_Avg = np.mean(CS_Real)
    CSReal_V_CSTheory = (CSReal_Avg - CSTheory_Avg)/CSTheory_Avg *100

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

def PlotParallel(column_groups,value_name,separate_firms=False):
    #Column Groups example:
    """
    column_groups = {
                "Monopoly": r"^MonopolyPrice_F",
                "Leader": r"^LeaderPrice_F",
                "Follower": r"^FollowerPrice_F",
            }
    """
    parquet_folder="TrainingResultsC"
    y_label = f"{value_name} Difference"
    title = f"{value_name} Differences (Data - Benchmark)"
    if separate_firms:
        save_path = f"TrainingResults/parallel_{value_name.lower()}_firm_boxplots.png"
    else:
        save_path = f"TrainingResults/parallel_{value_name.lower()}_boxplots.png"

    colours = {
            "Monopoly": "#C62828",
            "Leader": "#1565C0",
            "Follower": "#2E7D32",
            "MonopolyL": "#8E24AA",
            "MonopolyF": "#FB8C00",
        }

    ParquetFolder = Path(parquet_folder)

    #Group files by tested parameter
    grouped_files = defaultdict(list)
    for file in ParquetFolder.glob("*.parquet"):
        file_name = file.name

        #Safety to make sure file matches naming convention
        if file_name.startswith("run_") and "_" in file_name:
            details = file_name.split("_")

            tested_parameter = details[1] #get parameter name and value 
            df = pd.read_parquet(file) #load parquet information into dataframe obj

            grouped_files[tested_parameter].append(df)

    final_data_dict = {}
    for param, df_list in grouped_files.items():
        final_data_dict[param] = pd.concat(df_list, ignore_index=True)

    def parameter_sort_key(parameter):
        number_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', parameter)
        if number_match:
            prefix = parameter[:number_match.start()]
            return (prefix, float(number_match.group()), parameter)
        return (parameter, float("inf"), parameter)

    parameter_order = sorted(final_data_dict.keys(), key=parameter_sort_key)

    plot_rows = []
    for key,curr_df in final_data_dict.items():
        for group_label, column_pattern in column_groups.items():
            if isinstance(column_pattern, str):
                matched_columns = curr_df.filter(regex=column_pattern)
            else:
                existing_columns = [column for column in column_pattern if column in curr_df.columns]
                matched_columns = curr_df.loc[:, existing_columns]
            matched_columns = matched_columns.loc[:, [column for column in matched_columns.columns if not column.endswith("_Unit")]]

            def result_unit(row_index, firm_column):
                unit_column = f"{firm_column}_Unit"
                if unit_column in curr_df.columns:
                    unit = curr_df.at[row_index, unit_column]
                    if pd.notna(unit):
                        return unit
                return "Unknown"

            if separate_firms:
                for firm_column in matched_columns.columns:
                    for row_index, value in matched_columns[firm_column].dropna().items():
                        plot_rows.append({
                            "Parameter": key,
                            "Group": group_label,
                            "Firm": firm_column,
                            "Unit": result_unit(row_index, firm_column),
                            value_name: value,
                        })
            else:
                for firm_column in matched_columns.columns:
                    for row_index, value in matched_columns[firm_column].dropna().items():
                        plot_rows.append({
                            "Parameter": key,
                            "Group": group_label,
                            "Unit": result_unit(row_index, firm_column),
                            value_name: value,
                        })

    combined_values = pd.DataFrame(plot_rows)

    if combined_values.empty:
        available_columns = sorted({
            column
            for curr_df in final_data_dict.values()
            for column in curr_df.columns
        })
        requested_patterns = ", ".join(str(pattern) for pattern in column_groups.values())
        preview_columns = ", ".join(available_columns[:25])
        raise ValueError(
            "No matching columns were found for the requested column groups. "
            f"Requested patterns: {requested_patterns}. "
            f"First available columns: {preview_columns}"
        )

    fig, axes = plt.subplots(
        nrows=len(column_groups),
        ncols=1,
        figsize=(12, 4 * len(column_groups) + 1),
        sharex=True,
        sharey=False,
    )
    if len(column_groups) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14)

    for ax, group_label in zip(axes, column_groups.keys()):
        group_values = combined_values[combined_values["Group"] == group_label]
        units = group_values["Unit"].dropna().unique()
        if len(units) == 1:
            if units[0] == "Percent":
                axis_label = f"{value_name} (% Difference)"
            elif units[0] == "Difference":
                axis_label = f"{value_name} Difference"
            else:
                axis_label = f"{value_name} ({units[0]})"
        elif len(units) > 1:
            axis_label = f"{value_name} (Mixed Units)"
        else:
            axis_label = y_label
        sns.boxplot(
            data=group_values,
            x="Parameter",
            y=value_name,
            hue="Firm" if separate_firms else None,
            order=parameter_order,
            color=None if separate_firms else colours.get(group_label, "#666666"),
            ax=ax,
        )
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.0, zorder=1)
        ax.text(
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
        if separate_firms:
            ax.legend(title="Firm", loc="best")
        ax.set_title(group_label, pad=5)
        ax.set_xlabel("")
        ax.set_ylabel(axis_label)

    axes[-1].set_xlabel("Tested Parameter")
    fig.subplots_adjust(top=0.93, hspace=0.20)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

def PlotWelfareParallel():
    parquet_folder="TrainingResultsC"
    save_path_CS = f"TrainingResults/parallel_CS_boxplots.png"
    save_path_M = f"TrainingResults/parallel_Innovation_boxplots.png"

    ParquetFolder = Path(parquet_folder)

    #Group files by tested parameter
    grouped_files = defaultdict(list)
    for file in ParquetFolder.glob("*.parquet"):
        file_name = file.name

        #Safety to make sure file matches naming convention
        if file_name.startswith("run_") and "_" in file_name:
            details = file_name.split("_")

            tested_parameter = details[1] #get parameter name and value 
            df = pd.read_parquet(file) #load parquet information into dataframe obj

            grouped_files[tested_parameter].append(df)

    final_data_dict = {}
    for param, df_list in grouped_files.items():
        final_data_dict[param] = pd.concat(df_list, ignore_index=True)

    def parameter_sort_key(parameter):
        number_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', parameter)
        if number_match:
            prefix = parameter[:number_match.start()]
            return (prefix, float(number_match.group()), parameter)
        return (parameter, float("inf"), parameter)

    parameter_order = sorted(final_data_dict.keys(), key=parameter_sort_key)

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

    # Create the Box Plot - consumer surplus
    plt.figure(figsize=(8, 6))
    sns.boxplot(
    x="Key", 
    y="Consumer Surplus", 
    data=df_melted_CS, 
    palette="Set2", 
    order=parameter_order,
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

    
    plt.title("Consumer Surplus Distribution Across Keys", fontsize=14)
    plt.xlabel("Key Type", fontsize=12)
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
    order=parameter_order,
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


    plt.title("Innovation Distribution Across Keys", fontsize=14)
    plt.xlabel("Key Type", fontsize=12)
    plt.ylabel("Benchmark Innovation Rate vs Market", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path_M, dpi=300, bbox_inches='tight')
    plt.close()

def PlotStrategyParallel():
    parquet_folder="TrainingResultsC"
    save_path_CS = f"TrainingResults/parallel_convergence_boxplots.png"
    save_path_M = f"TrainingResults/parallel_states_boxplots.png"

    ParquetFolder = Path(parquet_folder)

    #Group files by tested parameter
    grouped_files = defaultdict(list)
    for file in ParquetFolder.glob("*.parquet"):
        file_name = file.name

        #Safety to make sure file matches naming convention
        if file_name.startswith("run_") and "_" in file_name:
            details = file_name.split("_")

            tested_parameter = details[1] #get parameter name and value 
            df = pd.read_parquet(file) #load parquet information into dataframe obj

            grouped_files[tested_parameter].append(df)

    final_data_dict = {}
    for param, df_list in grouped_files.items():
        final_data_dict[param] = pd.concat(df_list, ignore_index=True)

    def parameter_sort_key(parameter):
        number_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', parameter)
        if number_match:
            prefix = parameter[:number_match.start()]
            return (prefix, float(number_match.group()), parameter)
        return (parameter, float("inf"), parameter)

    parameter_order = sorted(final_data_dict.keys(), key=parameter_sort_key)

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

    # Create the Box Plot - consumer surplus
    plt.figure(figsize=(8, 6))
    sns.boxplot(
    x="Key", 
    y="Convergence Time", 
    data=df_melted_strategy, 
    palette="Set2", 
    order=parameter_order,
    hue="Key",      
    legend=False        
)
    
    plt.title("Market Convergence Time", fontsize=14)
    plt.xlabel("Key Type", fontsize=12)
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
    order=parameter_order,
    hue="Key",      
    legend=False        
)

    plt.title("Number of Unique States", fontsize=14)
    plt.xlabel("Key Type", fontsize=12)
    plt.ylabel("Unique States", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path_M, dpi=300, bbox_inches='tight')
    plt.close()

def Parallel_Outputs(Firms):
    #Normal plot
    PlotParallel(column_groups={"Monopoly": r"^MonopolyPrice_F","Leader": r"^LeaderPrice_F","Follower": r"^FollowerPrice_F",},
                value_name = "Prices")
    PlotParallel(column_groups={"Monopoly": r"^MonopolyInvest_F","Leader": r"^LeaderInvest_F","Follower": r"^FollowerInvest_F",},
                value_name = "Investments")
    PlotParallel(column_groups={"MonopolyL": r"^MonopolyLProfit_F","MonopolyF": r"^MonopolyFProfit_F","Leader": r"^LeaderProfit_F","Follower": r"^FollowerProfit_F"},
                value_name = "Profits")
    PlotParallel(column_groups={"MonopolyL": r"^MonopolyLMrktShr_F","MonopolyF": r"^MonopolyFMrktShr_F","Leader": r"^LeaderMrktShr_F","Follower": r"^FollowerMrktShr_F",},
                value_name = "MarketShares")
        
    #Leader Plot
    for leader in range(Firms):
        PlotParallel(column_groups={"Monopoly": f"^Leader{leader}_MonopolyPrice_F","Leader": f"^Leader{leader}_LeaderPrice_F","Follower": f"^Leader{leader}_FollowerPrice_F",},
                    value_name = f"Prices_Leader{leader}",
                    separate_firms=True)
        PlotParallel(column_groups={"Monopoly": f"^Leader{leader}_MonopolyInvest_F","Leader": f"^Leader{leader}_LeaderInvest_F","Follower": f"^Leader{leader}_FollowerInvest_F",},
                    value_name = f"Investments_Leader{leader}",
                    separate_firms=True)
        PlotParallel(column_groups={"MonopolyL": f"^Leader{leader}_MonopolyLProfit_F","MonopolyF": f"^Leader{leader}_MonopolyFProfit_F","Leader": f"^Leader{leader}_LeaderProfit_F","Follower": f"^Leader{leader}_FollowerProfit_F"},
                    value_name = f"Profits_Leader{leader}",
                    separate_firms=True)
        PlotParallel(column_groups={"MonopolyL": f"^Leader{leader}_MonopolyLMrktShr_F","MonopolyF": f"^Leader{leader}_MonopolyFMrktShr_F","Leader": f"^Leader{leader}_LeaderMrktShr_F","Follower": f"^Leader{leader}_FollowerMrktShr_F",},
                value_name = f"MarketShares_Leader{leader}",
                separate_firms=True)



    #Welfare Plot
    PlotWelfareParallel()

    #Strategy Plot
    PlotStrategyParallel()
