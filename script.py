import os

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
    os.makedirs(HISTORY_FILE_PATH, exist_ok=True)
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


def resample_data(data_frame, frequency='1W'):
    return data_frame.resample(frequency, closed='right', label='right').agg({'open': 'first',
                                                                              'high': 'max',
                                                                              'low': 'min',
                                                                              'close': 'last'}).dropna()


def get_max_min(prices, smoothing, window_range, column='close'):
    smooth_prices = prices['close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_max_dt.append(prices.iloc[i - window_range:i + window_range][column].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_min_dt.append(prices.iloc[i - window_range:i + window_range][column].idxmin())
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min.index.name = 'date'
    max_min = max_min.reset_index()
    # Nos quedamos con los no duplicados
    max_min = max_min[~max_min.date.duplicated()]
    p = prices.reset_index()
    max_min['day_num'] = p[p['timestamp'].isin(max_min.date)].index.values
    max_min = max_min.set_index('day_num')[column]

    return max_min


def get_min(prices, smoothing, window_range, column='close'):
    smooth_prices = prices['close'].rolling(window=smoothing).mean().dropna()
    local_min = argrelextrema(smooth_prices.values, np.less)[0]

    price_local_min_dt = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_min_dt.append(prices.iloc[i - window_range:i + window_range][column].idxmin())

    minima = pd.DataFrame(prices.loc[price_local_min_dt]).sort_index()
    minima.index.name = 'date'
    minima = minima.reset_index()
    # Nos quedamos con los no duplicados
    minima = minima[~minima.date.duplicated()]
    p = prices.reset_index()
    minima['day_num'] = p[p['timestamp'].isin(minima.date)].index.values
    minima = minima.set_index('day_num')[column]

    return minima


def get_max(prices, smoothing, window_range, column='close'):
    smooth_prices = prices['close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_max_dt.append(prices.iloc[i - window_range:i + window_range][column].idxmax())

    maxima = pd.DataFrame(prices.loc[price_local_max_dt]).sort_index()
    maxima.index.name = 'date'
    maxima = maxima.reset_index()
    # Nos quedamos con los no duplicados
    maxima = maxima[~maxima.date.duplicated()]
    p = prices.reset_index()
    maxima['day_num'] = p[p['timestamp'].isin(maxima.date)].index.values
    maxima = maxima.set_index('day_num')[column]

    return maxima


def get_stock(dataframe):
    return dataframe['close']


def RSI(data, time_window):
    diff = data.diff(1).dropna()  # diff in one field(one day)

    # this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]

    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def get_extreme_point(series, index):
    length = len(series)
    if index < 0 or index >= length:
        raise Exception("Error, wrong index of extreme point")
    return series.index[index], series.values[index]


def get_all_segments(series):
    length = len(series)
    result_array = []
    for index1 in range(length - 1):
        for index2 in range(index1 + 1, length):
            result_array.append([get_extreme_point(series, index1), get_extreme_point(series, index2)])
    return result_array


def plot_single_segment(segment):
    point1 = segment[0]
    point2 = segment[1]

    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values)


def plot_segments(segments):
    for segment in segments:
        plot_single_segment(segment)


def points_segment_cross_series(segment, series, is_max=True):
    init_index = segment[0][0]
    end_index = segment[1][0]
    # calculate function
    linear_equation = get_linear_equation_from_segment(segment)
    for i in range(init_index + 1, end_index):
        linear_equation_value = linear_equation(i)
        if (series.iloc[i] > linear_equation_value) and is_max:
            return True
        elif (series.iloc[i] < linear_equation_value) and (not is_max):
            return True
    return False


def get_linear_equation_from_segment(segment):
    slope = (segment[1][1] - segment[0][1]) / (segment[1][0] - segment[0][0])
    independent_variable = segment[0][1] - (slope * segment[0][0])
    return lambda x: (slope * x) + independent_variable


def filter_uncrossed_segments(segments, series, is_max):
    result_segments = []
    for segment in segments:
        if not points_segment_cross_series(segment, series, is_max):
            result_segments.append(segment)
    return result_segments


data = get_data('AAPL')
resampled_data = resample_data(data)

smoothing = 3
window = 10

max = get_max(resampled_data, smoothing, window)
min = get_min(resampled_data, smoothing, window)

all_segments_max = get_all_segments(max)
all_segments_min = get_all_segments(min)

filtered_segments_max = filter_uncrossed_segments(all_segments_max, resampled_data['close'], True)
filtered_segments_min = filter_uncrossed_segments(all_segments_min, resampled_data['close'], False)

resampled_data['RSI'] = RSI(resampled_data['close'], 14)

max_rsi = get_max(resampled_data, smoothing, window, "RSI")
min_rsi = get_min(resampled_data, smoothing, window, "RSI")

all_segments_max_rsi = get_all_segments(max_rsi)
all_segments_min_rsi = get_all_segments(min_rsi)

filtered_segments_max_rsi = filter_uncrossed_segments(all_segments_max_rsi, resampled_data['RSI'], True)
filtered_segments_min_rsi = filter_uncrossed_segments(all_segments_min_rsi, resampled_data['RSI'], False)


plt.subplot(2, 1, 1)
plt.plot()
resampled_data.reset_index()['close'].plot()
plt.scatter(max.index, max.values, color='orange', alpha=.5)
plt.scatter(min.index, min.values, color='green', alpha=.5)
plot_segments(filtered_segments_max)
plot_segments(filtered_segments_min)

plt.subplot(2, 1, 2)
resampled_data.reset_index()['RSI'].plot()
plt.scatter(max_rsi.index, max_rsi.values, color='orange', alpha=.5)
plt.scatter(min_rsi.index, min_rsi.values, color='green', alpha=.5)
plot_segments(filtered_segments_max_rsi)
plot_segments(filtered_segments_min_rsi)

plt.show()
