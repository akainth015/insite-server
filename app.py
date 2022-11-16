from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("connect")
def connected():
    print("client connected")
    emit("logs", {"data": "Connected"})


@app.route("/hooks/<node_id>")
def activate_web_hook(node_id):
    print("The web-hook for " + node_id + " was activated")
    socketio.emit("web-hook", (node_id, request.get_json(silent=True)))
    return "Success"


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
