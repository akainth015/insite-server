from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Models.LinearRegression import LinearRegressionTrainer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global array of models
models = {}


@socketio.on("linear")
def handle_linear_regression(nodeId, features, labels, feature_names, label_name):
    # Convert the features and feature names to dataframe
    features_df = pd.DataFrame(features, columns=feature_names)

    # Convert the labels to numpy array
    labels = np.array(labels)

    #Split the data into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(features_df, labels, test_size=0.2)

    #Create a LinearRegressionTrainer object
    trainer = LinearRegressionTrainer(len(features_df.columns) - 1,
                                        learning_rate=0.5,
                                        num_epochs=10000)



    #Train
    trainer.train(x_train, y_train, x_val, y_val)

    #Calculate Loss
    val_loss = trainer.evaluate(x_val, y_val)
    train_loss = trainer.evaluate(x_train, y_train)

    #Calculate Accuracy
    val_accuracy = trainer.accuracy(x_val, y_val)
    train_accuracy = trainer.accuracy(x_train, y_train)

    #Store train and val losses in result dictionary
    result = {
        "nodeId": nodeId,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }

    #Store the model in the global models dictionary
    models[nodeId] = trainer

    emit("linear", result, broadcast=False)


if __name__ == '__main__':
    socketio.run(app)