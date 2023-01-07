import datetime
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import mysql.connector as database
from sqlalchemy import create_engine


## Helper functions ##
def plot_result(df, xcol, ycol):
    df = df[df.columns[1:]]
    df.plot(x=xcol, y=ycol)
    plt.show()


def plot_gridsearch(df, bps, tojpg):
    df['avg_price'] = (df['High'] + df['Low'])/2

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['ADX_14'],
        name="ADX",
        line=dict(color="#1f77b4")
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['avg_price'],
        name="Average Price",
        yaxis="y2",
        line=dict(color="#000000")
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['revenue'],
        name="Revenue",
        yaxis="y3",
        line=dict(color="#d62728")
    ))

    '''
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['revenue'],
        name="Profit",
        yaxis="y4",
        line=dict(color="blue")
    ))'''

    '''
, np.full(
              shape=len(df),
              fill_value=25,
              dtype=int
            )
'''

    # Create axis objects
    fig.update_layout(
        xaxis=dict(
            title="Date"
        ),
        yaxis=dict(
            title="ADX",
            titlefont=dict(
                color="#1f77b4"
            ),
            tickfont=dict(
                color="#1f77b4"
            )
        ),
        yaxis2=dict(
            title="Avg. Prices",
            titlefont=dict(
                color="#000000"
            ),
            tickfont=dict(
                color="#000000"
            ),
            overlaying="y",
            side="left"
        ),
        yaxis3=dict(
            title="Revenue",
            titlefont=dict(
                color="#d62728"
            ),
            tickfont=dict(
                color="#d62728"
            ),
            overlaying="y",
            side="right"
        )
    )
    ''',
    yaxis4=dict(
        title="Profit",
        titlefont=dict(
            color="blue"
        ),
        tickfont=dict(
            color="blue"
        ),
        overlaying="y",
        side="right"
    )'''


    # Update layout properties
    fig.update_layout(
        title_text=f"Grid Trading with ADX under 25 and a Grid of {bps} Basis Point(s)",
        #width=1200,
    )

    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
    if tojpg == True:
        file_name = f"{len(df)}_datapoints_grid({bps})bps_trading_1.png"
        fig.write_image(file_name, format="png")

def establish_grid(baseprice, gridspacing):
    gridspacing = gridspacing / 10000
    buylines = np.arange(baseprice-(0.05+gridspacing), baseprice-gridspacing, 0.0001)
    sellines = np.arange(baseprice + gridspacing, baseprice+0.05+gridspacing, 0.0001)
    buylines, sellines = buylines.round(decimals=5), sellines.round(decimals=5)
    return buylines, sellines

def buy(buypositions, dfindex, balance, buylinescrossed, lastbuyprice, bps):
    maxCapital = 10000  # max capital to have invested at one point
    amountPerTrade = 100  # capital amount taken to buy/sell per trade
    for i in range(buylinescrossed):
        price = round(lastbuyprice + (buylinescrossed-i-1)*bps/10000, 4)
        #if balance < maxCapital:
         #   balance = balance - amountPerTrade
        buypositions = pd.concat([buypositions,  pd.DataFrame([[dfindex, price]], columns=['dfindex', 'price'])], axis=0, ignore_index=True)
    return buypositions

def sell(sellpositions, dfindex, balance, selllinescrossed, lastbuyprice, bps):
    maxCapital = 10000  # max capital to have invested at one point
    amountPerTrade = 100  # capital amount taken to buy/sell per trade
    for i in range(selllinescrossed):
        price = round(lastbuyprice - (selllinescrossed-i-1)*bps/10000, 4)
        #if balance < maxCapital:
         #   balance = balance - amountPerTrade
        sellpositions = pd.concat([sellpositions, pd.DataFrame([[dfindex, price]], columns=['dfindex', 'price'])], axis=0, ignore_index=True)
    return sellpositions

def check_value_exist(test_dict, value):
    do_exist = False
    for key, val in test_dict.items():
        if key == value:
            do_exist = True
    return do_exist

def sorttxhist(tx_hist):
    tx_hist = tx_hist.sort_values(by=['dfindex', 'price'], ascending=[True, False], ignore_index=True)
    tx_hist = (tx_hist
               .sort_values('dfindex')
               .assign(sort1=tx_hist['dfindex'].ne(tx_hist['dfindex'].shift(1)).cumsum())
               .assign(sort2=tx_hist['price'].mask(tx_hist['action'] == "buy", -tx_hist['price']))
               .sort_values(['sort1', 'sort2'])
               )
    #tx_hist.to_csv('tx_hist.csv')
    tx_hist = tx_hist[['dfindex', 'price', 'action']]
    return tx_hist

def liquidation(openorders, price, bps,liq_threshold, liquidations):
    pricerange = np.arange(price - (bps/10000 * liq_threshold), price + (bps/10000 * liq_threshold), 0.0001)
    pricerange = pricerange.round(decimals=4)

    dellist = []
    for i in range(len(openorders)):
        if openorders[i] not in pricerange and openorders[i] not in pricerange*(-1):
            dellist.append(openorders[i])

    for f in range(len(dellist)):
        openorders.remove(dellist[f])
        liquidations.append(dellist[f])

    return openorders, liquidations

def equalize_positions(buypositions, sellpositions, liq_threshold, bps, value):
    buypositions['action'] = 'buy'
    sellpositions['action'] = 'sell'
    tx_hist = pd.concat([buypositions, sellpositions], axis=0, ignore_index=True)
    tx_hist = sorttxhist(tx_hist)
    tx_hist['profit'] = 0
    tx_hist['revenue'] = 0
    openorders = []
    liquidations = []
    profit = 0

    oldindex = 0

    for i in range(len(tx_hist)):
        if tx_hist['action'].iloc[i] == 'buy':
            dellist = []
            dels = 0
            x = 0
            if openorders:
                while x < (len(openorders) - dels):
                    if tx_hist['price'].iloc[i] < -openorders[x]:
                            profit = profit + value
                            dellist.append(openorders[x])
                            openorders.remove(openorders[x])
                            dels = dels + 1
                            if -tx_hist['price'].iloc[i] not in openorders:
                                openorders.append(tx_hist['price'].iloc[i])
                    x = x + 1

            if not dellist:
                openorders.append(tx_hist['price'].iloc[i])

        else: # Sell
            dellist = []
            dels = 0
            x = 0
            if openorders:
                while x < (len(openorders) - dels):
                    if tx_hist['price'].iloc[i] > openorders[x] > 0:
                        profit = profit + value
                        dellist.append(openorders[x])
                        openorders.remove(openorders[x])
                        dels = dels + 1
                        if tx_hist['price'].iloc[i] not in openorders and openorders:
                            openorders.append(-tx_hist['price'].iloc[i])
                    x = x + 1

            if not dellist:
                openorders.append(-tx_hist['price'].iloc[i])

        openorders, liquidations = liquidation(openorders, tx_hist['price'].iloc[i], bps, liq_threshold, liquidations)
        expenses = len(openorders) * value + len(liquidations) * value
        if oldindex < tx_hist['dfindex'].iloc[i]:
            tx_hist['profit'].iloc[i] = profit
            tx_hist['revenue'].iloc[i] = profit-expenses
            oldindex = tx_hist['dfindex'].iloc[i]

    expenses = len(openorders)*value  + len(liquidations)*value
    revenue = profit-expenses
    #tx_hist.to_csv('txfinsis.csv')
    return tx_hist, profit, expenses, revenue



class Algorithms():

    def __init__(self):
        self.data = self.download_data()

    def download_data(self):
        '''Downloads and transforms the data for other functions'''
        first_data_date = '2018-10-28'
        today = date.today()
        resp = requests.get(
            f"YOUR-API-URL{first_data_date}&to_time={today}")  # change to variable timeframe at some point
        df = resp.json()
        df = pd.DataFrame.from_dict(df)
        df['Close'] = df['close']
        df['Open'] = df['open']
        df['High'] = df['high']
        df['Low'] = df['low']
        df['Date'] = pd.to_datetime(df['time'])
        df = df[["Open", "High", "Low", "Close", "Date"]]
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df = df.round(decimals=4)
        return df

    def select_window(self, windowfrom, windowto):
        data = self.data
        if windowfrom == False:
            windowfrom = 0
        if windowto == False:
            windowto = 0
        data = data[windowfrom:windowto]
        self.data = data

    def adx(self, plot=False):
        ## ADX ##
        df = self.data
        adx = ta.adx(high=df['High'], low=df['Low'], close=df['Close'], open=df['Open'], length=14)
        data = pd.concat([df, adx], axis=1)
        if plot == True:
            plot_result(data, 'Date', ["Open", "High", "Low", "Close", "ADX_14"])
        return data

    def moving_avrg(self, plot=False):
        df = self.data
        ## Moving Average ##
        mvngavrg = ta.alma(close=df['Close'], length=14)
        data = pd.concat([df, mvngavrg], axis=1)
        if plot == True:
            plot_result(data, 'Date', ['Open', 'Close', 'ALMA_14_6.0_0.85'])
        return data

    def bollinger_bands(self, plot=False, printresult=False):
        df = self.data
        ## Bollinger Bands ##
        bb = ta.bbands(close=df['Close'])
        data = pd.concat([df, bb], axis=1)
        data['bollinger_up'] = data['BBL_5_2.0']
        data['bollinger_down'] = data['BBM_5_2.0']

        in_market = False
        # calculate buy/sell signal
        data["signal"] = "hold"
        for tick in range(len(data)):
            if not in_market:
                if data['Close'].iloc[tick] < data['bollinger_down'].iloc[tick]:
                    if printresult == True:
                        print("buy")
                    in_market = True
                    data['signal'].iloc[tick] = "buy"
            elif in_market:
                if data['Close'].iloc[tick] > data['bollinger_up'].iloc[tick]:
                    if printresult == True:
                        print("sell")
                    in_market = False
                    data['signal'].iloc[tick] = "sell"

        if plot == True:
            plot_result(data, 'Date', ['Open', 'bollinger_up', 'bollinger_down'])
        return data

    def grid_search_with_adx(self, plot=False, printresult=False, tocsv=False, grid_spacing=10, liq_threshold=6, graphToJPG=False):
        data = self.adx()

        ## Presets
        balance = 0
        #grid_spacing = 10  # grid spacing in bps
        #liq_threshold = 6  # how many buy/selllines away the position get liquidated
        value = 100  # amount to long/short
        print('The Grid Search Algorythm has started!')

        ## Usage Objects
        buypositions = pd.DataFrame(columns=['dfindex', 'price'])
        sellpositions = pd.DataFrame(columns=['dfindex', 'price'])
        w = -1
        while w < len(data) - 1:
            w = w + 1
            if data['ADX_14'].iloc[w] <= 25:
                i = w
                buylines, sellines = establish_grid(data['Open'].iloc[i], grid_spacing)
                while data['ADX_14'].iloc[i] <= 25 and i < len(data) - 1:
                    if data['Close'].iloc[i] in buylines:
                        buylinescrossed = len(buylines) + 1 - np.where(buylines == data['Close'].iloc[i])[0][0]
                        buypositions = buy(buypositions, i, balance, buylinescrossed, data['Close'].iloc[i], grid_spacing)
                        # print(f"B{i}, Buy! At Closing price {data['Close'].iloc[i]} for the {buylinescrossed} time")
                        baseprice = data['Close'].iloc[i]
                        buylines, sellines = establish_grid(baseprice, grid_spacing)
                        i = i + 1

                        if data['Open'].iloc[i] in buylines:
                            buylinescrossed = len(buylines) + 1 - np.where(buylines == data['Open'].iloc[i])[0][0]
                            buypositions = buy(buypositions, i, balance, buylinescrossed, data['Open'].iloc[i],
                                               grid_spacing)
                            # print(f"BB{i}, Buy! At Opening price {data['Close'].iloc[i]} for the {buylinescrossed} time")
                            baseprice = data['Open'].iloc[i]
                            buylines, sellines = establish_grid(baseprice, grid_spacing)

                        elif data['Open'].iloc[i] in sellines:
                            selllinescrossed = 1 + np.where(sellines == data['Open'].iloc[i])[0][0]
                            sellpositions = sell(sellpositions, i, balance, selllinescrossed, data['Open'].iloc[i],
                                                 grid_spacing)
                            # print(f"BS{i}, Sell! At Opening price {data['Close'].iloc[i]} for the {selllinescrossed} time")
                            baseprice = data['Open'].iloc[i]
                            buylines, sellines = establish_grid(baseprice, grid_spacing)

                        else:
                            if printresult == True:
                                print('No line crossed', data['Open'].iloc[i])



                    elif data['Close'].iloc[i] in sellines:
                        selllinescrossed = 1 + np.where(sellines == data['Close'].iloc[i])[0][0]
                        sellpositions = sell(sellpositions, i, balance, selllinescrossed, data['Close'].iloc[i],
                                             grid_spacing)
                        # print(f"S{i}, Sell! At Closing price {data['Close'].iloc[i]} for the {selllinescrossed} time")
                        baseprice = data['Close'].iloc[i]
                        buylines, sellines = establish_grid(baseprice, grid_spacing)

                        i = i + 1

                        if data['Open'].iloc[i] in buylines:
                            buylinescrossed = len(buylines) + 1 - np.where(buylines == data['Open'].iloc[i])[0][0]
                            buypositions = buy(buypositions, i, balance, buylinescrossed, data['Open'].iloc[i],
                                               grid_spacing)
                            # print(f"BB{i}, Buy! At Opening price {data['Close'].iloc[i]} for the {buylinescrossed} time")
                            baseprice = data['Open'].iloc[i]
                            buylines, sellines = establish_grid(baseprice, grid_spacing)


                        elif data['Open'].iloc[i] in sellines:
                            selllinescrossed = 1 + np.where(sellines == data['Open'].iloc[i])[0][0]
                            sellpositions = sell(sellpositions, i, balance, selllinescrossed, data['Open'].iloc[i],
                                                 grid_spacing)
                            # print(f"BS{i}, Sell! At Opening price {data['Close'].iloc[i]} for the {selllinescrossed} time")
                            baseprice = data['Open'].iloc[i]
                            buylines, sellines = establish_grid(baseprice, grid_spacing)


                        else:
                            if printresult == True:
                                print('No line crossed', data['Open'].iloc[i])

                    else:
                        # print('No line crossed', data['Close'].iloc[i])
                        i = i + 1
                        if data['Open'].iloc[i] in buylines:
                            buylinescrossed = len(buylines) + 1 - np.where(buylines == data['Open'].iloc[i])[0][0]
                            buypositions = buy(buypositions, i, balance, buylinescrossed, data['Open'].iloc[i],
                                               grid_spacing)
                            # print(f"-B{i}, Buy! At Opening price {data['Close'].iloc[i ]} for the {buylinescrossed} time")
                            baseprice = data['Open'].iloc[i]
                            buylines, sellines = establish_grid(baseprice, grid_spacing)

                        elif data['Open'].iloc[i] in sellines:
                            selllinescrossed = 1 + np.where(sellines == data['Open'].iloc[i])[0][0]
                            sellpositions = sell(sellpositions, i, balance, selllinescrossed, data['Open'].iloc[i],
                                                 grid_spacing)
                            # print(f"-S{i}, Sell! At Opening price {data['Close'].iloc[i]} for the {selllinescrossed} time")
                            baseprice = data['Open'].iloc[i]
                            buylines, sellines = establish_grid(baseprice, grid_spacing)

                        else:
                            if printresult == True:
                                print('No line crossed', data['Open'].iloc[i])

                period = 0
                for i in range(len(data.loc[w:i])):
                    if data['ADX_14'].iloc[w] <= 25:
                        period = period + 1

                w = w + period

        tx_hist, profit, expenses, revenue = equalize_positions(buypositions, sellpositions, liq_threshold, grid_spacing,
                                                                value)
        tx_hist = tx_hist[tx_hist['profit'] != 0]
        profitdata = pd.merge(data, tx_hist, left_index=True, right_on='dfindex', how='outer')
        del profitdata['index'], profitdata['dfindex']
        profitdata = profitdata.reset_index()
        lastrev = 0
        lastprofit = 0
        for i in range(len(profitdata)):
            currentprice = profitdata['profit'].iloc[i]
            currentrev = profitdata['revenue'].iloc[i]
            if currentprice > 0:
                lastprofit = currentprice
                lastrev = currentrev
            else:
                profitdata['profit'].iloc[i] = lastprofit
                profitdata['revenue'].iloc[i] = lastrev

        profitdata['grid_size'] = grid_spacing
        if tocsv == True:
            profitdata.to_csv('profitdata.csv')
        if printresult==True:
            print('Profit:', revenue)
            print('Revenue:', profit)
            print('Cost of Profit', expenses)
        if plot == True:
            plot_gridsearch(profitdata, grid_spacing, graphToJPG)
        return profitdata

    def keltner_channel(self, plot=False):
        df = self.data
        ## Keltner Channel ##
        kc = ta.kc(high=df['High'], low=df['Low'], close=df['Close'], open=df['Open'], length=20)
        data = pd.concat([df, kc], axis=1)
        data['action'] = 'NaN'

        for i in range(len(data)):
            if data['KCLe_20_2'].iloc[i] > data['Close'].iloc[i]:
                data['action'].iloc[i] = 'buy'
            if data['KCUe_20_2'].iloc[i] < data['Close'].iloc[i]:
                data['action'].iloc[i] = 'sell'
        #data.to_csv('KC.csv')

        if plot == True:
            plot_result(data, 'Date', ['Open', 'Close', 'KCLe_20_2', 'KCBe_20_2', 'KCUe_20_2'])
        return data

    def hull_exponential_moving_average(self, plot=False):
        df = self.data
        ## Hull Exponential Moving Average ##
        hma = ta.hma(close=df['Close'])
        data = pd.concat([df, hma], axis=1)
        if plot == True:
            plot_result(data, 'Date', ['Close', 'HMA_10'])
        return data

    def moving_average_convergence_divergence(self, plot=False):
        df = self.data
        ## Moving Average Convergence Divergence ##
        macd = ta.macd(close=df['Close'], slow=21, fast=8)
        data = pd.concat([df, macd], axis=1)
        if plot == True:
            plot_result(data, 'Date', ['MACD_8_21_9', 'MACDh_8_21_9', 'MACDs_8_21_9'])
        return data

    def upload_to_DB(self, data, name, if_exists='append'):
        data = data.drop(['index','High', 'Low', 'Open', 'Close'], axis=1)
        print(data.columns)
        # create sqlalchemy engine
        engine = create_engine("mysql+mysqlconnector://{user}:{pw}@173.249.30.118/{db}"
                               .format(user="root",
                                       pw="mypass",
                                       db="IT_Project"))

        data.to_sql(name, con=engine, if_exists=if_exists)
        sql = f"SELECT * FROM {name}"
        result = engine.execute(sql).fetchall()
        columns = list(data.columns)
        result = pd.DataFrame(result, columns=columns.insert(0, 'index'))
        print(result)

    def show_DB_tables(self, printresult=True):
        connection = database.connect(
            user="root",
            password="PW", #replace with your own
            host="HOST", #replace with your own
            database="DB" #replace with your own
        )
        mycursor = connection.cursor()
        mycursor.execute("SHOW TABLES;")
        result = mycursor.fetchall()
        if printresult == True:
            print(result)
        connection.close()
        return result




