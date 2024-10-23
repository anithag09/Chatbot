import requests
import pandas as pd
from datetime import datetime

def fetch_stock_data(symbol, api_key):
    base_url = "https://www.alphavantage.co/query"
    function = "TIME_SERIES_DAILY"
    
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "full"
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        print(f"Error fetching data for {symbol}")
        return None
    
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame(time_series).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    
    return df

# Usage 
api_key = "ONQJX4JV2K8B7G7D"
symbol = "RELIANCE.BSE"  
stock_data = fetch_stock_data(symbol, api_key)

if stock_data is not None:
    print(f"Stock data for {symbol}:")
    print(stock_data.head())
    
    # Save to CSV
    stock_data.to_csv(f"{symbol}_stock_data.csv")
    print(f"Data saved to {symbol}_stock_data.csv")