from flask import Flask
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS

app = Flask(__name__)
# socketio = SocketIO(app)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("connect")
def connected():
    print("client connected")
    emit("logs", {"data": "Connected"})


@socketio.on("new_node")
def new_node_added(node_data):
    print("data:" + str(node_data))
    emit("logs", {"data": node_data})


@socketio.on("default_node")
def default_node():
    emit("logs", {"data": "default_node"})


if __name__ == '__main__':
    # socketio.run(app, debug=True, port=3000)
    socketio.run(app)
"""
@app.route("/")
def hello():
  return "Hello World!"

if __name__ == "__main__":
  app.run()
"""
