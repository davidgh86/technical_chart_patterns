import sys

import config
import constants

import os
import pandas as pd
from matplotlib import pyplot as plt
import alpaca_trade_api as tradeapi
import numpy as np
from scipy.signal import argrelextrema
import datetime

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


def get_max_min(prices, smoothing_extremes, window_range, column='close'):
    smooth_prices = prices['close'].rolling(window=smoothing_extremes).mean().dropna()
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


def get_minimums(prices, smoothing_extremes, window_range, column='close'):
    smooth_prices = prices['close'].rolling(window=smoothing_extremes).mean().dropna()
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


def get_maximums(prices, smoothing_extremes, window_range, column='close'):
    smooth_prices = prices['close'].rolling(window=smoothing_extremes).mean().dropna()
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


def RSI(data_frame, time_window):
    diff = data_frame.diff(1).dropna()  # diff in one field(one day)

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
    parameters = get_linear_equation_parameters(segment)
    return lambda x: (parameters[0] * x) + parameters[1]


def get_linear_equation_parameters(segment):
    slope = (segment[1][1] - segment[0][1]) / (segment[1][0] - segment[0][0])
    independent_variable = segment[0][1] - (slope * segment[0][0])
    return slope, independent_variable


def filter_uncrossed_segments(segments, series, is_max):
    result_segments = []
    for segment in segments:
        if not points_segment_cross_series(segment, series, is_max):
            result_segments.append(segment)
    return result_segments


def matches_time(price_segment, rsi_segment, offset=0):
    matches_start = abs(price_segment[0][0] - rsi_segment[0][0]) <= offset
    if not matches_start:
        return False
    return abs(price_segment[1][0] - rsi_segment[1][0]) <= offset


def decreases(segment):
    return segment[0][1] > segment[1][1]


def increases(segment):
    return segment[0][1] < segment[1][1]


def is_divergent_or_convergent(price_segment, rsi_segment, offset=0):
    if matches_time(price_segment, rsi_segment, offset):
        if decreases(price_segment) and increases(rsi_segment):
            return True
        if increases(price_segment) and decreases(rsi_segment):
            return True
    return False


def get_diff(segment, series):
    equation_function = get_linear_equation_from_segment(segment)
    diff_array = []
    for index in range(segment[0][0] + 1, segment[1][0]):
        segment_y = equation_function(index)
        diff_array.append(series.iloc[index] - segment_y)
    return diff_array


def get_directional_relationship(price_segment, price_series, rsi_segment, rsi_series):
    linear_equation_price_parameters = get_linear_equation_parameters(price_segment)
    price_slope = linear_equation_price_parameters[0]
    linear_equation_rsi_parameters = get_linear_equation_parameters(rsi_segment)
    rsi_slope = linear_equation_rsi_parameters[0]
    directional_relationship_type = "divergent" if price_slope > 0 else "convergent"
    price_diff = get_diff(price_segment, price_series)
    price_area = np.sum(price_diff)
    rsi_diff = get_diff(rsi_segment, rsi_series)
    rsi_area = np.sum(rsi_diff)

    index_segments = price_segment[0][0], price_segment[1][0] + 1

    filtered_price_series = price_series.iloc[index_segments[0]:index_segments[1]]
    filtered_rsi_series = rsi_series.iloc[index_segments[0]:index_segments[1]]

    max_min_index_label = filtered_price_series.idxmax() if price_area > 0 else filtered_price_series.idxmin()
    index_absolute_position_max_min = price_series.index.get_loc(max_min_index_label)
    index_relative_position_max_min = filtered_price_series.index.get_loc(max_min_index_label)

    equation_segment = get_linear_equation_from_segment(price_segment)
    segment_value_in_max_min = equation_segment(index_absolute_position_max_min)

    filtered_price_series_min = filtered_price_series.min()
    filtered_price_series_max = filtered_price_series.max()

    extremes_type = "min" if price_area > 0 else "max"

    if filtered_rsi_series[0] > 70:
        rsi_entry_range = "over_buy"
        if rsi_slope < 0 and extremes_type == "max":
            valid_segment = True
            tendency = "decreasing"
        else:
            valid_segment = False
            tendency = "unknown"
    elif filtered_rsi_series[0] < 30:
        rsi_entry_range = "over_sell"
        if rsi_slope > 0 and extremes_type == "min":
            valid_segment = True
            tendency = "rising"
        else:
            valid_segment = False
            tendency = "unknown"
    else:
        rsi_entry_range = "hidden"
        if price_slope > 0 and extremes_type == "min":
            valid_segment = True
            tendency = "rising"
        elif price_slope < 0 and extremes_type == "max":
            valid_segment = True
            tendency = "decreasing"
        else:
            valid_segment = False
            tendency = "unknown"

    if extremes_type == "max":
        height_extreme_segment = segment_value_in_max_min - filtered_price_series_min
    else:
        height_extreme_segment = filtered_price_series_max - segment_value_in_max_min

    cross_chart_value = filtered_price_series_min if extremes_type == "max" else filtered_price_series_max

    limit_size_relative_index = index_relative_position_max_min * (1 + constants.FIBONACCI_VALUE)

    if valid_segment and limit_size_relative_index <= filtered_price_series.size:
        valid_segment = False

    if valid_segment:
        activation_price = cross_chart_value
        if tendency == "rising":
            buying_price = cross_chart_value
            selling_price = cross_chart_value + height_extreme_segment
        elif tendency == "decreasing":
            buying_price = cross_chart_value - height_extreme_segment
            selling_price = cross_chart_value
        else:
            valid_segment = False
    else:
        activation_price = None
        buying_price = None
        selling_price = None

    return {
        "valid": valid_segment,
        "price_segment": price_segment,
        "rsi_segment": rsi_segment,
        "extremes_type": extremes_type,
        "directional_relationship_type": directional_relationship_type,
        "slope_abs_diff": abs(price_slope) + abs(rsi_slope),
        "price_relationship_info": {
            "slope": price_slope,
            "area": abs(price_area),
            "min": filtered_price_series_min,
            "max": filtered_price_series_max,
            "mean": filtered_price_series.mean(),
            "standard_deviation": filtered_price_series.std(),
            "analytics_indicator_info": {
                "tendency": tendency,
                "rsi_entry_range": rsi_entry_range,
                "cross_chart_value": cross_chart_value,
                "max_min_index_label": max_min_index_label,
                "index_absolute_position_max_min": index_absolute_position_max_min,
                "index_relative_position_max_min": index_relative_position_max_min,
                "segment_value_in_max_min": segment_value_in_max_min,
                "height_extreme_segment": height_extreme_segment,
                "activation_price": activation_price,
                "buying_price": buying_price,
                "selling_price": selling_price
            },
            "diff": {
                "sum": price_area,
                "min": np.min(price_diff),
                "max": np.max(price_diff),
                "mean": np.mean(price_diff),
                "standard_deviation": np.std(price_diff)
            }
        },
        "rsi_relationship_info": {
            "slope": rsi_slope,
            "area": abs(rsi_area),
            "min": filtered_rsi_series.min(),
            "max": filtered_rsi_series.max(),
            "mean": filtered_rsi_series.mean(),
            "standard_deviation": filtered_rsi_series.std(),
            "diff": {
                "sum": rsi_area,
                "min": np.min(rsi_diff),
                "max": np.max(rsi_diff),
                "mean": np.mean(rsi_diff),
                "standard_deviation": np.std(rsi_diff)
            }
        }
    }


def join_segments(price_segments, price_series, rsi_segments, rsi_series):
    valid_convergences_divergences = []
    for price_segment in price_segments:
        for rsi_segment in rsi_segments:
            if is_divergent_or_convergent(price_segment, rsi_segment):
                valid_convergences_divergences.append(
                    get_directional_relationship(price_segment, price_series, rsi_segment, rsi_series))
    return valid_convergences_divergences


number_of_arguments = len(sys.argv)
if number_of_arguments != 2 and number_of_arguments != 3 and number_of_arguments != 5:
    print("python.py symbol ([temporality] | temporality [start-date end-date]) ")
    print("example python.py AAPL")
    print("example python.py AAPL 1W")
    print("example python.py AAPL 1W 29-3-2018:15:27 5-4-2019-08:15:27")
    exit()

data = get_data(sys.argv[1])
if len(sys.argv) > 2:
    resampled_data = resample_data(data, sys.argv[2])
else:
    resampled_data = resample_data(data)

if len(sys.argv) > 3:
    start_date_string = sys.argv[3]
    end_date_string = sys.argv[4]
    try:
        start_date = datetime.datetime.strptime(start_date_string, '%d-%m-%Y-%H:%M:%S')
        end_date = datetime.datetime.strptime(end_date_string, '%d-%m-%Y-%H:%M:%S')
        if start_date >= end_date:
            print("La fecha inicial tiene que ser anterior a la final")
            exit()
        resampled_data = resampled_data.loc[start_date:end_date]
    except ValueError:
        print("El formato de fechas debe de ser del tipo 9-6-2019-8:15:27 con precisión máxima de segundos")
        exit()

smoothing = 3
window = 10

maximums = get_maximums(resampled_data, smoothing, window)
minimums = get_minimums(resampled_data, smoothing, window)

all_segments_max = get_all_segments(maximums)
all_segments_min = get_all_segments(minimums)

filtered_segments_max = filter_uncrossed_segments(all_segments_max, resampled_data['close'], True)
filtered_segments_min = filter_uncrossed_segments(all_segments_min, resampled_data['close'], False)

resampled_data['RSI'] = RSI(resampled_data['close'], 14)

max_rsi = get_maximums(resampled_data, smoothing, window, "RSI")
min_rsi = get_minimums(resampled_data, smoothing, window, "RSI")

all_segments_max_rsi = get_all_segments(max_rsi)
all_segments_min_rsi = get_all_segments(min_rsi)

filtered_segments_max_rsi = filter_uncrossed_segments(all_segments_max_rsi, resampled_data['RSI'], True)
filtered_segments_min_rsi = filter_uncrossed_segments(all_segments_min_rsi, resampled_data['RSI'], False)

max_divergences_segments = join_segments(filtered_segments_max, resampled_data['close'],
                                         filtered_segments_max_rsi, resampled_data['RSI'])
min_divergences_segments = join_segments(filtered_segments_min, resampled_data['close'],
                                         filtered_segments_min_rsi, resampled_data['RSI'])

divergences = max_divergences_segments + min_divergences_segments

prices_divergences = []
rsi_divergences = []
for divergence in divergences:
    prices_divergences.append(divergence["price_segment"])
    rsi_divergences.append(divergence["rsi_segment"])

plt.subplot(2, 1, 1)
plt.plot()
resampled_data.reset_index()['close'].plot()
plt.scatter(maximums.index, maximums.values, color='orange', alpha=.5)
plt.scatter(minimums.index, minimums.values, color='green', alpha=.5)
plot_segments(prices_divergences)
# plot_segments(filtered_segments_min)

plt.subplot(2, 1, 2)
resampled_data.reset_index()['RSI'].plot()
plt.scatter(max_rsi.index, max_rsi.values, color='orange', alpha=.5)
plt.scatter(min_rsi.index, min_rsi.values, color='green', alpha=.5)
plot_segments(rsi_divergences)
# plot_segments(filtered_segments_min_rsi)


plt.show()
