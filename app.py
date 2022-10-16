from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def hello():
  return "Hello World!"

if __name__ == "__main__":
  app.run()
