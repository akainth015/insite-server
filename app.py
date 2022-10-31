from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("linear")
def handle_linear_regression(json):
    print("Received a json " + str(type(json)))

    #Drop the category called "y" and store it in a matrix called y
    y = json["y"]
    df_y = pd.read_json(y, orient='split')

    #Store the other categories in a matrix called x
    x = json.drop(columns=["y"])
    df_x = pd.read_json(x, orient='split')

    #Split the data into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(df_x,
                                                      df_y,
                                                      test_size=0.2,
                                                      random_state=42)

    print(x_train)
    print(y_train)

    #Create a LinearRegressionTrainer object
    trainer = LinearRegressionTrainer(dataset.columns.size - 1,
                                      learning_rate=0.5,
                                      num_epochs=10000)
    emit("linear", json, broadcast=False)


if __name__ == '__main__':
    socketio.run(app)