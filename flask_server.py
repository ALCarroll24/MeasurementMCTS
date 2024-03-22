from flask import Flask, request, render_template_string
from threading import Thread
import requests
import ctypes

class FlaskServer:
    node_clicked = None
    
    def __init__(self):
        self.flask_thread = None

    @staticmethod
    def create_app():
        app = Flask(__name__)

        @app.route('/')
        def index():
            # Assuming your Pyvis HTML content is saved as `graph.html` in the templates directory
            return render_template_string(open('tree_visualization.html').read())

        @app.route('/node_click', methods=['POST'])
        def node_click():
            data = request.json
            print(data)
            node_id = data['nodeId']
            FlaskServer.node_clicked = int(node_id)
            print(f"Node clicked: {node_id}")
            print(f"Node clicked @ class: {FlaskServer.node_clicked}")
            # Here you can add any logic you want to process the clicked node
            return {"status": "success"}
        
        return app
        
    def stop_flask(self):
        # Make a POST request to the /shutdown route
        # requests.post('http://localhost:5000/shutdown')
        
        # Force kill the thread
        self.async_raise(self.flask_thread, SystemExit)
        
    
    def run(self):
        app = FlaskServer.create_app()
        app.run(debug=False, use_reloader=False)
    
    def run_thread(self):
        self.flask_thread = Thread(target=self.run)
        self.flask_thread.start()
        
    def async_raise(self, thread_obj, exctype):
        """Raises an exception in the threads with id tid"""
        if not thread_obj.is_alive():
            raise ValueError("Thread is not active")
        
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_obj.ident), ctypes.py_object(exctype)
        )
        if res == 0:
            raise ValueError("Invalid thread ID")
        elif res > 1:
            # If it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_obj.ident, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

if __name__ == "__main__":
    app = FlaskServer.create_app()
    app.run(debug=True)