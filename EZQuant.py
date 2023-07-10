import datetime
import json
import keyboard
import pytz
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from backtesting import Backtest, Strategy
import warnings
from twilio.rest import Client
from fredapi import Fred

import secretkeys


def monitor_stock(period):
    rfr = get_risk_free_rate()
    print("1. Single Stock")
    print("2. Market")
    print()
    action = input("> ")
    print("_____________________________________________________")
    action = int(action)
    if action == 1:
        tick = input("Enter Stock/ETF ticker: ")
        tick = tick.upper()
        print("_____________________________________________________")
        print("Beginnning monitor for " + tick.upper() + ", press 'q' to quit.")
        first = True
        buy_sig = False
        sell_sig = False
        action = "Hold"
        last_price = 0.0
        account_sid = secretkeys.TWILIO_ACCOUNT_SID
        auth_token = secretkeys.TWILIO_AUTH_TOKEN
        client = Client(account_sid, auth_token)
        while True:
            if keyboard.is_pressed('q'):
                print("Quitting...")
                break
            hist = download_stock_history(tick, 0)
            hist = prepare_strategy(hist, period[0], period[1], period[2], period[3])
            bt = Backtest(hist, ShortingStrategy, cash=500, commission=0)
            stats = bt.run()
            trades = bt.run()._trades
            rets = trades['ReturnPct']
            for idx, trade in trades.iterrows():
                trades.at[idx, 'ReturnPct'] *= 100
            last_row = hist.iloc[-1][['Buy', 'Sell', 'RSI', 'MFI']]
            live_price = hist.iloc[-1]['Open']
            if first:
                last_price = float(live_price)
            momentum = float(live_price) - float(last_price)
            momentum = round(momentum, 3)
            print(f"Live price: {live_price}")
            print()
            print(last_row)
            print()
            print(f"Momentum: {momentum}")
            if buy_sig != last_row['Buy'] or sell_sig != last_row['Sell'] or first:
                buy_sig = last_row['Buy']
                sell_sig = last_row['Sell']
                if first:
                    first = False
                if buy_sig and not sell_sig:
                    action = "Buy"
                elif not buy_sig and sell_sig:
                    action = "Sell"
                else:
                    action = "Hold"
                # Set environment variables for your credentials
                # Read more at http://twil.io/secure
                message = client.messages.create(
                    body=action + " " + str(tick) + ", Momentum: $" + str(momentum) + ".",
                    from_=secretkeys.TWILIO_FROM_NUM,
                    to=secretkeys.TWILIO_TO_NUM
                )
                print(f"Sent message to {secretkeys.TWILIO_TO_NUM}, ID: ", message.sid)
            calc_sortino(rfr, stats['Return [%]'], rets)
            print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print("_____________________________________________________")
            last_price = float(live_price)
            time.sleep(30)


def short_sell_bot(period):
    print("1. Single Stock")
    print("2. Market")
    print()
    action = input("> ")
    action = int(action)
    print("_____________________________________________________")
    if action == 1:
        tick = input("Enter Stock/ETF ticker: ")
        hist = download_stock_history(tick, period[4])
        hist = prepare_strategy(hist, period[0], period[1], period[2], period[3])
        bt = Backtest(hist, ShortingStrategy, cash=500, commission=0)
        stats = bt.run()
        print(stats)
        trades = bt.run()._trades
        rets = trades['ReturnPct']
        for idx, trade in trades.iterrows():
            trades.at[idx, 'ReturnPct'] *= 100
        rfr = get_risk_free_rate()
        calc_sortino(rfr, stats['Return [%]'], rets)
        # for i, trade in trades.iterrows():
        #     print(f'Trade {i}:')
        #     print(trade)
        #     print('---')
        warnings.filterwarnings('ignore')
        bt.plot()
    elif action == 2:
        print("1. S&P 500")
        print("2. VOLAT")
        print("3. SOX")
        print("4. Special")
        print("5. Portfolio")
        print("6. Extended Portfolio")
        print("7. BANK")
        print("8. ALPHA")
        print()
        action = input("> ")
        print("_____________________________________________________")
        num = int(action)
        df = test_against_market(ShortingStrategy, period, num)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        print(len(df))


def get_end_date():
    # Get the US stock market calendar
    nyse = mcal.get_calendar('NYSE')

    # Get the current datetime in Eastern Time (US stock market time)
    now = datetime.datetime.now(pytz.timezone('America/New_York'))

    # Get the market schedule for the last 30 days
    start_date = now.date() - datetime.timedelta(days=30)
    schedule = nyse.schedule(start_date=start_date, end_date=now.date())

    # Check if the market is open today
    market_open_today = now.date() in schedule.index

    if market_open_today:
        # If the market is open today, use today as the current date
        current_date = now.date()
    else:
        # If the market is closed today, use the last business day as the current date
        current_date = schedule.index[-1].date()

    # print(f"Current date: {current_date}")
    return current_date


def get_dates(day_count):
    # Get the US stock market calendar
    end_date = get_end_date()
    nyse = mcal.get_calendar('NYSE')

    # Get the market schedule for the last (n+30) days
    start_date = end_date - datetime.timedelta(days=(day_count + 30))
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    # Find the closest market open date within n regular days before the given date
    market_date = None
    dates = schedule.index.date
    current_date = end_date - datetime.timedelta(days=1)
    for i in range(1, day_count + 1):
        # print("i",i)
        # print("Current date", current_date)
        # print(dates)
        # print(current_date in dates)
        if current_date in dates:
            start_date = current_date
            # print("Start date", start_date)
            break
        current_date = end_date - datetime.timedelta(days=i)
        # print("New date", current_date)

    # start_date = market_date
    print(f"Current date: {end_date}")
    print(f"Closest market open date within {day_count} regular days before current date: {start_date}")
    return start_date, end_date


# Downloads stock data and returns history in a pandas dataframe
def download_stock_history(ticker, period):
    # Define the date range for the data you want to fetch
    stock = yf.Ticker(ticker)
    if period == 0:
        # start_date = datetime.datetime(2023, 5, 4)
        # end_date = datetime.datetime(2023, 5, 3)
        start_date, end_date = get_dates(1)
        # start_date += datetime.timedelta(days=1)
        end_date += datetime.timedelta(days=1)
        interval = "1m"
        data = stock.history(start=start_date, end=end_date, interval=interval)
        return data
    elif period == 1:
        # start_date = datetime.datetime(2023, 4, 21)
        # end_date = datetime.datetime.now(2023, 5, 7)
        start_date, end_date = get_dates(30)
        interval = "1h"
        data = stock.history(start=start_date, end=end_date, interval=interval)
        return data


# Calculating simple moving average (SMA)
def calculate_sma(data, window):
    sma = data['Close'].rolling(window=window).mean()
    return sma


# Calculating exponential moving average (EMA)
def calculate_ema(data, window):
    ema = data['Close'].ewm(span=window).mean()
    return ema


# Calculating relative strength index (RSI)
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Calculating moving average converegence divergence (MACD)
def calculate_macd(data):
    ema_12 = calculate_ema(data, 12)
    ema_26 = calculate_ema(data, 26)
    macd = ema_12 - ema_26
    return macd


# Calculating money flow index
def calculate_mfi(data, period=14):
    # Calculate the Typical Price for each day
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3

    # Calculate the Money Flow for each day
    money_flow = typical_price * data['Volume']

    # Calculate the Positive Money Flow and Negative Money Flow
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    # Calculate the Money Flow Ratio
    positive_money_flow_sum = positive_money_flow.rolling(window=period).sum()
    negative_money_flow_sum = negative_money_flow.rolling(window=period).sum()
    money_flow_ratio = positive_money_flow_sum / negative_money_flow_sum

    # Calculate the Money Flow Index
    mfi = 100 - (100 / (1 + money_flow_ratio))
    # print(mfi)
    return mfi


# Prepare buy/sell strategy for backtesting
def prepare_strategy(data, short, med, long, rsi, sim=False):
    # Prepare data
    ema_short = calculate_ema(data, short)
    ema_med = calculate_ema(data, med)
    ema_long = calculate_ema(data, long)
    rsi = calculate_rsi(data, rsi)
    mfi = calculate_mfi(data)
    data['EMA_SHORT'] = ema_short
    data['EMA_MED'] = ema_med
    data['EMA_LONG'] = ema_long
    data['RSI'] = rsi
    data['MFI'] = mfi
    prices = data['Close']
    data['Buy'] = False
    data['Sell'] = False
    # print(data.index[0])
    # print("0000-04-20 09:30:00-04:00")
    # Moving Average Crossover Strategy
    for idx, row in data.iterrows():
        # Uptrend: Price above all EMAs
        if row['Close'] > row['EMA_SHORT'] and row['Close'] > row['EMA_MED'] and row['Close'] > row['EMA_LONG']:
            data.at[idx, 'Buy'] = True
        # Downtrend: Price below all EMAs
        elif row['Close'] < row['EMA_SHORT'] and row['Close'] < row['EMA_MED'] and row['Close'] < row['EMA_LONG']:
            data.at[idx, 'Sell'] = True

        # Potential reversal signals
        if idx > data.index[0]:
            # Long position: 10 EMA crosses above 30 EMA, and 30 EMA is above 50 EMA
            if row['EMA_SHORT'] > row['EMA_MED'] > row['EMA_LONG'] and row.shift(1)['EMA_SHORT'] <= row.shift(1)[
                'EMA_MED']:
                data.at[idx, 'Buy'] = True

            # Short position: 10 EMA crosses below 30 EMA, and 30 EMA is below 50 EMA
            elif row['EMA_SHORT'] < row['EMA_MED'] < row['EMA_LONG'] and row.shift(1)['EMA_SHORT'] >= row.shift(1)[
                'EMA_MED']:
                data.at[idx, 'Sell'] = True

            # Longer-term trend reversal signals
            # Downtrend to uptrend: 10 EMA crosses above 30 EMA, and 30 EMA is below 50 EMA
            elif row['EMA_SHORT'] > row['EMA_MED'] and row.shift(1)['EMA_SHORT'] <= row.shift(1)['EMA_MED'] and row[
                'EMA_MED'] < row['EMA_LONG']:
                data.at[idx, 'Buy'] = True

            # Uptrend to downtrend: 10 EMA crosses below 30 EMA, and 30 EMA is above 50 EMA
            elif row['EMA_SHORT'] < row['EMA_MED'] and row.shift(1)['EMA_SHORT'] >= row.shift(1)['EMA_MED'] and row[
                'EMA_MED'] > row['EMA_LONG']:
                data.at[idx, 'Sell'] = True

    # RSI Overbuy/Oversell Strategy
    # data['RSI_Signal'] = 0
    # data.loc[data['RSI'] < 30, 'RSI_Signal'] = 1
    # data.loc[data['RSI'] > 71.12, 'RSI_Signal'] = -1
    # for idx, row in data.iterrows():
    #     if row['Buy'] and row['RSI_Signal'] == -1:
    #         data.at[idx, 'Buy'] = False
    #     elif row['Sell'] and row['RSI_Signal'] == 1:
    #         data.at[idx, 'Sell'] = False

    # MFI Overbuy/Oversell Strategy
    data['MFI_Signal'] = 0
    data.loc[data['MFI'] < 20, 'MFI_Signal'] = 1
    data.loc[data['MFI'] > 80, 'MFI_Signal'] = -1
    for idx, row in data.iterrows():
        if row['Buy'] and row['MFI_Signal'] == -1:
            data.at[idx, 'Buy'] = False
        elif row['Sell'] and row['MFI_Signal'] == 1:
            data.at[idx, 'Sell'] = False
    # print(data[['Close', 'Buy', 'Sell']].head(60))
    # print()

    # Simulate buy/sell pattern
    if sim:
        count = 4
        num_shares = 0
        bal = 500
        legacy_bal = 500
        sell = True
        start = True
        for idx, row in data.iterrows():
            if row['Sell'] and start:
                sell = False
                continue
            close_price = str(row['Close'])
            if row['Buy'] and (not sell):
                price = count * float(row['Close'])
                bal -= price
                print("Bought at $" + close_price)
                print("Price: " + str(price))
                print("Balance: " + str(bal))
                print()
                sell = True
                num_shares += count
            elif row['Sell'] and sell:
                price = count * float(row['Close'])
                bal += price
                print("Sold at $" + close_price)
                print("Price: " + str(price))
                print("Balance: " + str(bal))
                print()
                sell = False
                num_shares -= count
        print()
        if num_shares > 0:
            price = data['Close'].iloc[-1] * num_shares
            bal += price
        performance = ((bal - legacy_bal) / legacy_bal) * 100
        print("Value: " + str(bal))
        print("Percent increase/decrease: " + str(performance))
    return data


def calc_returns(data):
    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Signal'].shift(1)

    # Calculate cumulative returns
    data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()

    print(data[['Close', 'Cumulative_Returns', 'Cumulative_Strategy_Returns']].tail(60))
    print(data[['Close', 'Returns', 'Strategy_Returns']].tail(60))


def remove_bad_stocks(index):
    ticker_lists = pd.read_json("tickers.json", typ="series").to_dict()
    tickers_to_remove = ticker_lists["tickers_to_remove"]
    result = [ticker for ticker in index if ticker not in tickers_to_remove]
    return result


def add_ticker_to_list(new_ticker, list_name="tickers_to_remove"):
    # Read the current JSON data
    with open("tickers.json", "r") as f:
        ticker_lists = json.load(f)

    # Add the new ticker to the specified list
    if list_name in ticker_lists:
        if new_ticker not in ticker_lists[list_name]:
            ticker_lists[list_name].append(new_ticker)
        else:
            print(f"Ticker {new_ticker} already exists in {list_name}.")
    else:
        print(f"List {list_name} not found.")

    # Write the updated JSON data back to the file
    with open("tickers.json", "w") as f:
        json.dump(ticker_lists, f, indent=2)


def get_risk_free_rate():
    # Replace 'your_api_key' with your actual FRED API key
    api_key = secretkeys.FRED_API_KEY
    fred = Fred(api_key=api_key)

    # Fetch the latest 13-week T-bill yield
    tbill_13_week_data = fred.get_series('DTB3')
    tbill_13_week_yield = tbill_13_week_data.iloc[-1]

    # Convert the annual yield to a daily yield
    risk_free_rate_daily = (1 + tbill_13_week_yield / 100) ** (1 / 365) - 1

    # print("13-week T-bill yield:", tbill_13_week_yield)
    # print("Daily risk-free rate:", risk_free_rate_daily)
    return risk_free_rate_daily * 7


def get_downside_dev(ret):
    neg_ret = ret[ret < 0]
    # print(neg_ret)
    downside_dev = np.sqrt((neg_ret ** 2).mean())
    return downside_dev


def calc_sortino(rfr, avg_ret, ret):
    # print("Average Return Per Trade: ", ret.mean())
    # print("Aggregate Return: ", avg_ret)
    # print("Risk-Free Rate: ", rfr)
    downside_dev = get_downside_dev(ret)
    # print("Std Downside Dev: ", downside_dev)
    sortino = (avg_ret - rfr) / downside_dev
    print("Sortino Ratio: ", sortino)
    return sortino


def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    sp500_df = table[0]
    sp500_tickers = sp500_df['Symbol'].tolist()
    return sp500_tickers


def test_against_market(Strategy, period, num):
    ticker_lists = pd.read_json("tickers.json", typ="series").to_dict()
    index = []
    if num == 1:
        index = remove_bad_stocks(get_sp500())
    elif num == 2:
        index = remove_bad_stocks(ticker_lists["volat"])
    elif num == 3:
        index = remove_bad_stocks(ticker_lists["sox"])
    elif num == 4:
        index = remove_bad_stocks(ticker_lists["special"])
    elif num == 5:
        index = remove_bad_stocks(ticker_lists["portfolio"])
    elif num == 6:
        index = remove_bad_stocks(ticker_lists["ext_portfolio"])
    elif num == 7:
        index = remove_bad_stocks(ticker_lists["bank"])
    elif num == 8:
        index = remove_bad_stocks(ticker_lists["alpha"])
    df = pd.DataFrame(index, columns=['Tickers'])
    df['Market Return'] = 0
    df['Strat Return'] = 0
    df['Sortino'] = 0
    total_return = 0
    total_bah_return = 0
    num_tickers = 0
    num_neg_tickers = 0
    num_pos_tickers = 0
    total_pos_ret = 0
    total_pos_bah_ret = 0
    total_neg_ret = 0
    total_neg_bah_ret = 0
    rfr = get_risk_free_rate()
    for ticker in index:
        try:
            print(str(num_tickers) + ": " + str(ticker))
            hist = download_stock_history(ticker, period[4])
            prepare_strategy(hist, period[0], period[1], period[2], period[3])
            bt = Backtest(hist, Strategy, cash=500, commission=0)
            stats = bt.run()
            strat_ret = stats['Return [%]']
            bah_ret = stats['Buy & Hold Return [%]']
            total_return += strat_ret
            total_bah_return += bah_ret
            if bah_ret < 0:
                num_neg_tickers += 1
                total_neg_ret += strat_ret
                total_neg_bah_ret += bah_ret
            elif bah_ret > 0:
                num_pos_tickers += 1
                total_pos_ret += strat_ret
                total_pos_bah_ret += bah_ret
            df.at[num_tickers, 'Strat Return'] = strat_ret
            df.at[num_tickers, 'Market Return'] = bah_ret
            trades = bt.run()._trades
            rets = trades['ReturnPct']
            for idx, trade in trades.iterrows():
                trades.at[idx, 'ReturnPct'] *= 100
            df.at[num_tickers, 'Sortino'] = calc_sortino(rfr, stats['Return [%]'], rets)
            num_tickers += 1
            print(stats)
            print()
            print("---------------------------------------------")
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Security {ticker} likely no longer exists, adding it to \"Deprecated Stocks\"")
            add_ticker_to_list(ticker)
            num_tickers += 1

    df = df.sort_values(by=['Market Return', 'Strat Return', 'Sortino'], ascending=True)
    print(df)
    print()
    avg_return = total_return / num_tickers
    avg_bah_return = total_bah_return / num_tickers
    print("Average Strategy Return: " + str(avg_return))
    print("Average Market Return: " + str(avg_bah_return))
    if num_pos_tickers > 0:
        avg_pos_return = total_pos_ret / num_pos_tickers
        avg_pos_bah_return = total_pos_bah_ret / num_pos_tickers
        print("---------------------------------------------")
        print("Average Strategy Return for Upward Moving Stocks: " + str(avg_pos_return))
        print("Average Market Return for Upward Moving Stocks: " + str(avg_pos_bah_return))
    if num_neg_tickers > 0:
        avg_neg_return = total_neg_ret / num_neg_tickers
        avg_neg_bah_return = total_neg_bah_ret / num_neg_tickers
        print("---------------------------------------------")
        print("Average Strategy Return for Downward Moving Stocks: " + str(avg_neg_return))
        print("Average Market Return for Downward Moving Stocks: " + str(avg_neg_bah_return))
    return df


# Backtesting class
class ShortingStrategy(Strategy):
    def init(self):
        self.buy_signal = self.I(lambda: self.data['Buy'], name='Buy')
        self.sell_signal = self.I(lambda: self.data['Sell'], name='Sell')

    def next(self):
        if self.sell_signal[-1] == 1:
            # Close existing short position before buying
            if self.position.is_long:
                self.position.close()
            self.sell()
        elif self.buy_signal[-1] == 1:
            # Close existing long position before selling
            if self.position.is_short:
                self.position.close()
            self.buy()


short_sell_bot([14, 30, 59, 6, 0])
