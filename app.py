from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on("linear_regression")
def handle_linear_regression(json):
    print('received json: ' + str(json))



if __name__ == '__main__':
    socketio.run(app)