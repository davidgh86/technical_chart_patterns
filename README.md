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

python script.py AAPL 1W 9-6-2014-8:15:27 9-6-2019-8:15:27
python script.py symbol temporality start-date end-date 

date format dd-mm-yyyy-hh:MM:ss
temporality parameters (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects)

xA -> X Years

xM -> X Months

xW -> x weeks

xD -> X days

xH -> X hours

xT -> X min