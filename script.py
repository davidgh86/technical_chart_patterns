import pandas as pd
import config

from matplotlib import pyplot as plt
import alpaca_trade_api as tradeapi
import numpy as np
from scipy.signal import argrelextrema

api = tradeapi.REST(config.API_KEY,
                    config.SECRET_KEY,
                    'https://paper-api.alpaca.markets')

HISTORY_FILE_PATH = 'historic_data'


def get_data_from_advantage(symbol):
    rest_data = api.alpha_vantage.historic_quotes(symbol)
    df = pd.DataFrame.from_dict(rest_data, orient="index")
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index.name = 'timestamp'
    df.drop(columns=['volume'], inplace=True)
    df.dropna(inplace=True)
    df = df[~df.index.duplicated()]
    df.replace(0, method='bfill', inplace=True)
    return df


def get_symbol_path(symbol):
    return HISTORY_FILE_PATH + "/" + symbol + ".csv"


def exists_history_data(symbol):
    try:
        f = open(get_symbol_path(symbol))
        exists = True
        f.close()
    except IOError:
        exists = False
    return exists


def get_data(symbol):
    if exists_history_data(symbol):
        df = load_csv(symbol)
    else:
        df = get_data_from_advantage(symbol)
        df.index = pd.to_datetime(df.index)
        df.to_csv(get_symbol_path(symbol))
        df = load_csv(symbol)
    return df


def load_csv(symbol):
    df = pd.read_csv(get_symbol_path(symbol), index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


data = get_data('AAPL')


def resample_data(data_frame, frequency='1W'):
    return data_frame.resample(frequency, closed='right', label='right').agg({'open': 'first',
                                                                              'high': 'max',
                                                                              'low': 'min',
                                                                              'close': 'last'}).dropna()


resampled_data = resample_data(data)


def get_max_min(prices, smoothing, window_range):
    smooth_prices = prices['close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_max_dt.append(prices.iloc[i - window_range:i + window_range]['close'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_min_dt.append(prices.iloc[i - window_range:i + window_range]['close'].idxmin())
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min.index.name = 'date'
    max_min = max_min.reset_index()
    # Nos quedamos con los no duplicados
    max_min = max_min[~max_min.date.duplicated()]
    p = prices.reset_index()
    max_min['day_num'] = p[p['timestamp'].isin(max_min.date)].index.values
    max_min = max_min.set_index('day_num')['close']

    return max_min


smoothing = 3
window = 10

minmax = get_max_min(resampled_data, smoothing, window)

resampled_data.reset_index()['close'].plot()
plt.scatter(minmax.index, minmax.values, color='orange', alpha=.5)
plt.show()
