from flask import Flask, request,jsonify
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
from yfinance import Ticker as TC
from yfinance import download as download
from datetime import date as dt
import pandas as pd
import yfinance as yf




app = Flask(__name__)
# socketio = SocketIO(app)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/http-call")
def http_call():
    """return JSON with string data as the value"""
    data = {'data':'This text was fetched using an HTTP call to server on render'}
    return jsonify(data)


@socketio.on("/")
def get_market_price(sid = None, data = None):
  msft_data = TC("MSFT")
  market_price = None
  for key, value in msft_data.info.items():
    if key == "regularMarketPrice":
      market_price = value 
      break
  socketio.send(market_price)
  return f'<h>{market_price}</h>'

@socketio.on("/get_historical")
def get_historical(sid = None, data = None):
  msft_data = TC("MSFT")
  parced_msft_data = []
  for value in msft_data.history(period = '10y'):
    parced_msft_data.append(value)      

  return f'<h>{parced_msft_data}</h>'



@socketio.on("/get_csv")
def get_chart_price():
  tickerStrings = ['MSFT']
  df_list = list()
  for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period='max')
    data['ticker'] = ticker  # add this column because the dataframe doesn't contain a column with the ticker
    df_list.append(data)

# combine all dataframes into a single dataframe
  df = pd.concat(df_list)
# save to csv
  df.to_csv('ticker.csv')
  return f'<h> hello </h>'





if __name__ == '__main__':
    socketio.run(app)
