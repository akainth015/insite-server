from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("linear")
def handle_linear_regression(json):
    print("Received a json " + str(type(json)))
    emit("linear", json, broadcast=False)


if __name__ == '__main__':
    socketio.run(app)