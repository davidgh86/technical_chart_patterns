Technical chart patterns analysis in stocks data series with alpaca, alphaadvantage and pandas.
this script works with alpaca sdk. in order to work you must configure APCA_API_KEY_ID, APCA_API_SECRET_KEY and 
ALPHAVANTAGE_API_KEY environment variables as explained here https://github.com/alpacahq/alpaca-trade-api-python/

Is also necessary add ANACONDA_HOME and ANACONDA_DIVERGENCIAS_ENV as environment variables 
ANACONDA_HOME is the Anaconda installation path usually C:\ProgramData\Anaconda3
ANACONDA_DIVERGENCIAS_ENV is and anaconda enviroment with the next dependencies installed
- pandas
- matplotlib
- alpaca_trade_api
- numpy
- scipy

usage:

python.py AAPL 2018-06-29-08:15:27.243860 2019-06-29-08:15:27.243860 1W
python.py symbol start-date end-date temporality