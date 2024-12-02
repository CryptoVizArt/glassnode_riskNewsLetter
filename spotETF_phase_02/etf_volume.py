import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import date, datetime, time, timedelta


today = datetime.combine(date.today(), time())

def stock_ticker(ticker):
    # Download historical financial data for AAPL
    response = yf.download(ticker, start="2010-01-01", end=today)
    response['date'] = pd.to_datetime(response.index,utc=True)
    response = response.set_index('date',drop=False)
    response.index.name = ''
    return response

def btc_etf_tickers():

    assets = [
        'GBTC','BTC','IBIT','FBTC','ARKB',
        'BITB','BTCO','HODL','BRRR',
        'EZBC','BTCW'
    ]
    response = pd.DataFrame(columns=['date'])
    count = 0
    for i in assets:
        etf = stock_ticker(i)
        vol_name    = i+'_Volume'
        price_name  = i+'_Price'
        if count == 0:
            response['date'] = etf['date']
            response['AGG_Volume'] = 0
        response[price_name] = etf['Close']
        response[vol_name]   = etf['Volume']*etf['Close']
        response[vol_name]   = response[vol_name].fillna(0)
        response['AGG_Volume'] = response['AGG_Volume'] + response[vol_name]
        count+=1
    
    filepath = ''
    response.to_csv(filepath+'etf_volume.csv')

    return response

df = btc_etf_tickers()