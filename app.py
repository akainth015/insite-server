from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

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




@socketio.on("linear")
def handle_linear_regression(nodeId, features, labels, feature_names, label_name):
    # Convert the features and feature names to dataframe
    features_df = pd.DataFrame(features, columns=feature_names)

    # Convert the labels to numpy array
    labels = np.array(labels)

    # Split the data into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(features_df, labels, test_size=0.2)

    # Create a linear regression model and fit
    model = LinearRegression().fit(x_train, y_train)

    # Get the predictions on the training and validation data
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # Get the training and validation accuracy
    train_accuracy = model.score(x_train, y_train)
    val_accuracy = model.score(x_val, y_val)
    

    # Get the training and validation loss
    train_loss = mean_squared_error(y_train, y_train_pred)
    val_loss = mean_squared_error(y_val, y_val_pred)


    #Store train and val losses in result dictionary
    result = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy
    }


    emit("linear", (nodeId, result), broadcast=False)

if __name__ == '__main__':
    # socketio.run(app, debug=True, port=3000)
    socketio.run(app)

