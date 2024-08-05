from graphviz import Digraph
import pickle
import argparse
import os
from pyvis.network import Network
import json
import numpy as np

def dir_path(string):
    if os.path.exists(string):
        return string
    else:
        raise NotADirectoryError(string)


def name(node):
    return node.__repr__().replace(") (", ")\n(")


def recursive_graph(node, g):

    def inner_fnct(node):
        g.node(str(node.__hash__()), name(node))
        for k, v in node.children.items():
            g.node(str(v.__hash__()), name(v), shape='square')
            g.edge(str(node.__hash__()), str(v.__hash__()))
            for k2, v2 in v.children.items():
                g.node(str(v2.__hash__()), label=name(v2))
                g.edge(str(v.__hash__()), str(v2.__hash__()))
                inner_fnct(v2)

    inner_fnct(node)

def render_graph(root, open=True):
    g = Digraph()
    recursive_graph(root, g)
    
    if open:
        g.render('img/tree.gv', view=True)
        
    print('Tree rendered in img/tree.gv')

def add_nodes_and_edges_pyvis(node, net, parent_hash=None):
    node_hash = str(node.__hash__())
    
    # If this is a random node, make it square
    if "RandomNode" in str(type(node)):
        # Check if mean reward can be calculated
        if node.visits > 0:
            mean_reward = node.cumulative_reward/(node.visits)
        else:
            mean_reward = 0
        # Build the text string label for the node
        label = "Action: " + " " + str(np.around(node.action, 2)) + "\n" + \
            "Mean Reward: " + " " + str(np.round(mean_reward, 2)) + "\n" + \
            "Cum Reward: " + " " + str(np.round(node.cumulative_reward, 2)) + "\n" + \
            "Visits: " + " " + str(np.round(node.visits, 2))
            
        # Find the node level (2 times the horizon step + 1 for the random node)
        horizon_step = node.parent.state[3]
        level = 2 * horizon_step + 1
        
        net.add_node(node_hash, label=label, shape='square', level=level)
    else:
        # Get corner covariance diagonals
        corner_covariance = np.diag(node.state[2])
        
        # Check if this is the root node
        if node.is_root:
            root_label = "Root\n"
        else:
            root_label = ""
        
        # Build the text string label for the node
        label = root_label + \
            "C1 Variance: " + " " + str(np.around(corner_covariance[0:2], 2)) + "\n" + \
            "C2 Variance: " + " " + str(np.around(corner_covariance[2:4], 2)) + "\n" + \
            "C3 Variance: " + " " + str(np.around(corner_covariance[4:6], 2)) + "\n" + \
            "C4 Variance: " + " " + str(np.around(corner_covariance[6:8], 2)) + "\n" + \
            "Reward: " + " " + str(np.round(node.reward, 2)) + "\n" + \
            "Visits: " + " " + str(np.round(node.visits, 2)) + "\n" + \
            "Evaluation Reward: " + " " + str(np.round(node.evaluation_reward, 2)) + "\n" + \
            "Horizon Step: " + " " + str(node.state[3])
            
        # Find the level (2 times the horizon step)
        level = 2 * node.state[3]
        net.add_node(node_hash, label=label, level=level)
    
    if parent_hash is not None:
        net.add_edge(parent_hash, node_hash)
    
    for child in node.children.values():
        add_nodes_and_edges_pyvis(child, net, node_hash)

def render_pyvis(root):
    net = Network(height="1200px", width="100%", directed=True)
    net.force_atlas_2based()
    add_nodes_and_edges_pyvis(root, net)
    
    ### Both show buttons and setting options don't work together
    # net.show_buttons()
    # Set the hierarchical layout options
    hierarchical_options  = {
      "layout": {
        "hierarchical": {
          "enabled": True,
          "levelSeparation": 700,
          "nodeSpacing": 180,
          "sortMethod": "directed" # hubsize, directed
        }
      },
      "physics": {
        "enabled": False,
      },
      "interaction": {
        "dragNodes": False,
      }
    }
    options_json = json.dumps(hierarchical_options)
    net.set_options(options_json)
    
    # Save and open the network graph in a browser
    net.save_graph("tree_visualization.html")

    # Add JavaScript code to the HTML file to handle node click events
    javascript_code = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        network.on("click", function (params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                fetch('/node_click', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({nodeId: nodeId}),
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Success:', data);
                })
                .catch((error) => {
                    console.error('Error:', error);
                });
            }
        });
    });
    </script>
    """
    insert_javascript("tree_visualization.html", javascript_code)


def insert_javascript(filepath, javascript_code):
    with open(filepath, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Specify where you want to insert your JavaScript code
    # Here, we're inserting it before the closing </body> tag
    insertion_point = html_content.rfind('</body>')
    modified_content = html_content[:insertion_point] + javascript_code + html_content[insertion_point:]
    
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(modified_content)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=dir_path)

    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        data = pickle.load(f)

    root = data["root"]

    g = Digraph()

    recursive_graph(root)
    file_name = os.path.basename(args.path)

    g.render('img/'+file_name+'.gv', view=True)
