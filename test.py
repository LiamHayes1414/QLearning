
import matplotlib.pyplot as plt
import networkx as nx# 1. Initialize a directional graph
G = nx.DiGraph()

# 2. Define nodes and their exact (x, y) grid coordinates
# Key: Node Name/Number -> Value: (X-coordinate, Y-coordinate)
pos = {
    "10": (0, 10),   # Root node
    "20": (1, 20),   # Upper branch Option 1
    "30": (1, 30),   # Lower branch Option 2
    "40": (2, 40),   # Sub-options...
    "50": (2, 50),
    "60": (2, 60)
}

# Add nodes to the graph
G.add_nodes_from(pos.keys())

# 3. Define arrows (edges) and their small labels
# Format: (from_node, to_node, label_text)
edges = [
    ("10", "20", "p=0.6"),
    ("10", "30", "p=0.4"),
    ("20", "40", "Option A"),
    ("20", "50", "Option B"),
    ("30", "50", "Option C"),
    ("30", "60", "Option D")
]

for u, v, label in edges:
    G.add_edge(u, v, label=label)

# 4. Configure visual styling
node_colors = ["#1f77b4" if n == "10" else "#ff7f0e" for n in G.nodes()]
fig, ax = plt.subplots(figsize=(8, 5))

# 5. Draw the network components
# Draw the nodes at their exact (x, y) coordinates
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)

# Draw the labels inside the nodes
nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", font_size=10, ax=ax)

# Draw the directional arrows between nodes
nx.draw_networkx_edges(
    G, pos, 
    arrowstyle="-|>", 
    arrowsize=15, 
    edge_color="gray", 
    width=2, 
    node_size=800, # Ensures arrows stop at node boundaries
    ax=ax
)

# Draw the small labels along the arrows
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="red", ax=ax)

# 6. Force the Matplotlib x and y axes to stay visible
ax.set_axis_on()
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.set_xlabel("Time Step (t)")
ax.set_ylabel("State/Value Level")
plt.grid(True, linestyle="--", alpha=0.5) # Optional background grid

plt.show()