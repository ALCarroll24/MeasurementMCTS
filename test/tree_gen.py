from pyvis.network import Network

# Function to add nodes and edges
def add_nodes_and_edges(net, node, level=0, parent=None):
    node_id = f"{node['name']}_{level}"
    net.add_node(node_id, label=node['name'], title=f"Node: {node['name']}")
    
    if parent:
        net.add_edge(parent, node_id)
    
    for child in node.get('children', []):
        add_nodes_and_edges(net, child, level + 1, node_id)

# Sample data: a simple tree
data = {
    "name": "Root",
    "children": [
        {"name": "Child 1", "children": [{"name": "Grandchild 1"}, {"name": "Grandchild 2"}]},
        {"name": "Child 2"},
        {"name": "Child 3", "children": [{"name": "Grandchild 3"}]}
    ]
}

# Create a network
net = Network()
net.force_atlas_2based()

# Add nodes and edges from the sample data
add_nodes_and_edges(net, data)

# Show the interactive graph
net.save_graph("test.html")
