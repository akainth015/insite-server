import os
import sys
import meaningcloud
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

from flask import Flask, request
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
from yfinance import Ticker as TC
from yfinance import download as download
import pandas as pd
import yfinance as yf
import csv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

model = 'IAB_en'
license_key = os.getenv('MEANING_CLOUD_TOKEN')

Sentiment_record = dict()
Lemma_record = dict()
Lang_record = dict()


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

@socketio.on("get_market_price")
def get_market_price(node_id, company_name):
    print(company_name)
    compInfo = TC(company_name)
    error = "Improper Ticker Symbol"
    for key, value in compInfo.info.items():
        if key == "regularMarketPrice":
            if(value != None):
                return value
            else:
                return error

@socketio.on("get_historical_prices")
def get_chart_price(node_id, company_name):
    df_list = list()
    data = yf.download(company_name, group_by="Ticker", period='1mo')
    data['ticker'] = company_name  # add this column because the dataframe doesn't contain a column with the ticker
    df_list.append(data)
    df = pd.concat(df_list)
    df.to_csv('ticker.csv')
    final_result = []

    with open("ticker.csv") as csvFile:
        reader = csv.DictReader(csvFile)
        for row in reader:
            final_result.append(row)
            
    return final_result


     

if __name__ == '__main__':
    socketio.run(app, debug=False, port=5000)
