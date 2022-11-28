import os
import sys
import meaningcloud
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

model = 'IAB_en'
license_key = os.getenv('MEANING_CLOUD_TOKEN')

Sentiment_record = dict()
Lemma_record = dict()
Lang_record = dict()
models_record = dict()


@socketio.on("connect")
def connected():
    print("client connected")
    emit("logs", {"data": "Connected"})


@app.route("/hooks/<node_id>")
def activate_web_hook(node_id):
    print("The web-hook for " + node_id + " was activated")
    socketio.emit("web-hook", (node_id, request.get_json(silent=True)))
    return "Success"


@socketio.on("new_node")
def new_node_added(node_data):
    print("data:" + str(node_data))
    emit("logs", {"data": node_data})


@app.route('/sentiment/<input_string>', methods=['POST'])
def sentiment(input_string):
    if input_string in Sentiment_record:
        return Sentiment_record[input_string]
    else:
        try:
            # SENTIMENT API CALL
            sentiment_response = meaningcloud.SentimentResponse(
                meaningcloud.SentimentRequest(license_key,
                                              lang='en',
                                              txt=input_string,
                                              txtf='plain').sendReq())

            Sentiment_record[input_string] = sentiment_response.getResponse()
            return Sentiment_record[input_string]

        except ValueError:
            e = sys.exc_info()[0]
            print("\nException: " + str(e))
            return ("\nException: " + str(e))


@app.route('/lemma/<input_string>', methods=['POST'])
def lemma(input_string):
    if input_string in Lemma_record:
        return Lemma_record[input_string]
    else:
        try:
            # Syntax API
            parser_response = meaningcloud.ParserResponse(
                meaningcloud.ParserRequest(license_key,
                                           txt=input_string,
                                           lang='en').sendReq())

            lemma_resp = "Lemmatization:\n"
            if parser_response.isSuccessful():
                lemmatization = parser_response.getLemmatization(True)
                # print(lemmatization)
                for token, analyses in lemmatization.items():
                    lemma_resp += ("\tToken ->" + token)
                    for analysis in analyses:
                        lemma_resp += ("\t\tLemma -> " + analysis['lemma'])
                        lemma_resp += ("\t\tPoS Tag ->" + analysis['pos'] +
                                       "\n")
            else:
                lemma_resp += "*Unable to find lemmatization for the input"
            Lemma_record[input_string] = dict(lemma=lemma_resp)
            return Lemma_record[input_string]

        except ValueError:
            e = sys.exc_info()[0]
            print("\nException: " + str(e))
            return ("\nException: " + str(e))


@app.route('/language/<input_string>', methods=['POST'])
def language(input_string):
    if input_string in Lang_record:
        return Lang_record[input_string]
    else:
        try:
            # Language Identification API
            lang_response = meaningcloud.LanguageResponse(
                meaningcloud.LanguageRequest(license_key,
                                             txt=input_string).sendReq())

            if lang_response.isSuccessful():
                first_lang = lang_response.getFirstLanguage()
                if first_lang:
                    language = lang_response.getLanguageCode(first_lang)
                    resp = "\tLanguage detected: " + \
                        lang_response.getLanguageName(
                            first_lang) + ' (' + language + ")\n"
                    Lang_record[input_string] = dict(language=resp)
                    return Lang_record[input_string]
                else:
                    return ("\tNo language detected!\n")

        except ValueError:
            e = sys.exc_info()[0]
            print("\nException: " + str(e))
            return ("\nException: " + str(e))


@socketio.on("linear")
def handle_linear_regression(nodeId, features, labels, feature_names,
                             label_name):
    # Convert the features and feature names to dataframe
    features_df = pd.DataFrame(features, columns=feature_names)

    # Convert the labels to numpy array
    labels = np.array(labels)

    # Split the data into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(features_df,
                                                      labels,
                                                      test_size=0.2)

    # Create a linear regression model and fit
    model = LinearRegression().fit(x_train, y_train)

    # Get the predictions on the training and validation data
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # Get the training and validation accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

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

    #Store the model in the global models dictionary
    models[nodeId] = trainer

    emit("linear", (nodeId, result), broadcast=False)


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
