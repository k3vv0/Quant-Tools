# Quantitative Finance Tools: EZQuant

## Introduction

Welcome to the EZQuant repository, your one-stop-shop for quantitative finance tools. The goal is to make financial analysis and algorithmic trading more accessible by providing easy-to-use, effective, and efficient tools for everyday use.

Currently, the primary feature is the EZQuant application which allows you to backtest any security or even entire markets against a collection of sophisticated trading algorithms. EZQuant makes backtesting as simple as it can possibly be. I am continuously developing and adding new features, stay tuned for more exciting tools!

## Installation and Dependencies

### Installing Dependencies

EZQuant utilizes several Python libraries for data analysis, finance, web scraping, and more. All dependencies are listed in the `requirements.txt` file in this repository. To install these dependencies, you'll want to use pip, a package manager for Python. You can install all necessary dependencies by running the following command in your terminal:

```sh
pip install -r requirements.txt
```
This will automatically install all the necessary libraries listed in the requirements.txt file.

### API Keys
For EZQuant to function correctly, you will need to supply it with a few API keys. These keys are stored in a Python file named keys.py. Currently, the API keys required are:

FRED_API_KEY : For fetching data from the Federal Reserve Economic Data (FRED) API.  
TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN : For sending SMS alerts with Twilio.

## Running the Application
To run the EZQuant application, navigate to the directory containing the application files in your terminal and then run the following command:

```sh
python EZQuant.py
```
You'll then see EZQuant's user interface, which will guide you through the process of selecting a security or market, choosing a trading algorithm, and backtesting it.

## Algorithms

The main algorithmic function `prepare_strategy()` is designed to apply various technical indicators to financial data and execute trading strategies based on these indicators.

The function takes as input a pandas DataFrame containing price data, and three time periods (`short`, `med`, and `long`) for calculating exponential moving averages (EMAs). It also takes a `rsi` period for calculating the Relative Strength Index (RSI), and a `sim` flag that determines whether to simulate trades based on the indicators.

We calculate EMAs of the closing prices for the `short`, `med`, and `long` periods, the RSI for the given `rsi` period, and the Money Flow Index (MFI). The calculated EMAs, RSI, and MFI values are then added as new columns to the DataFrame.

The trading logic within the function can be divided into two main strategies:

1. **Moving Average Crossover Strategy**: We iterate through the DataFrame and mark 'Buy' points whenever the closing price is above all three EMAs, indicating an uptrend. Conversely, 'Sell' points are marked whenever the closing price is below all three EMAs, indicating a downtrend. Additional conditions are added to account for potential reversal signals. We consider both the short term (where the `short` EMA crosses above or below the `med` EMA) and longer-term trend reversals (where the `short` EMA crosses the `med` EMA while the `med` EMA is below or above the `long` EMA).

2. **Overbought/Oversold Strategy**: Here, the MFI is used as an indicator of overbought or oversold conditions. If a 'Buy' point is marked and the MFI indicates an overbought condition (above 80), the 'Buy' point is invalidated. Similarly, if a 'Sell' point is marked and the MFI indicates an oversold condition (below 20), the 'Sell' point is invalidated.

If the `sim` flag is set to True, the function will simulate a simple trading strategy using a set starting balance. It will 'buy' or 'sell' a fixed number of shares based on the 'Buy' and 'Sell' signals, and will calculate the final account balance and performance of the strategy.

By combining these two strategies, EZQuant aims to capture both trends and reversals in the market, and to avoid trading in overbought or oversold conditions. Please note that this is a very simplistic simulation and does not take into account trading fees or taxes, nor does it simulate any form of risk management. Always use caution when backtesting and never rely solely on backtesting for trading decisions.


_Note: This project is meant for educational and informational purposes only. It should not be considered financial advice. Always do your own research before making any investment decisions._
