from flask import Flask, render_template_string, request
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Assuming your Pyvis HTML content is saved as `graph.html` in the templates directory
    return render_template_string(open('tree_visualization.html').read())

@app.route('/node_click', methods=['POST'])
def node_click():
    data = request.json
    node_id = data['nodeId']
    print(f"Node clicked: {node_id}")
    # Here you can add any logic you want to process the clicked node
    return {"status": "success"}

if __name__ == "__main__":
    app.run(debug=True)
