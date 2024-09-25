import pandas as pd
import os
import numpy as np

# Function to read stock data from an Excel file
def get_stock_data_from_excel(file_path):
    all_stock_data = pd.read_excel(file_path, sheet_name=None)
    stock_data = {}

    for sheet_name, data in all_stock_data.items():
        # Ensure 'Date' is treated as a datetime column and set as index
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        stock_data[sheet_name] = data

    return stock_data

# Function to calculate EMA for a given stock's dataframe
def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

# Function to calculate Stochastic Oscillator
def calculate_stochastic(data, period_k, period_d, period_smooth):
    lowest_low = data['Low'].rolling(window=period_k).min()
    highest_high = data['High'].rolling(window=period_k).max()
    k_percent = (data['Close'] - lowest_low) * 100 / (highest_high - lowest_low)
    k_percent_smooth = k_percent.rolling(window=period_smooth).mean()
    d_percent = k_percent_smooth.rolling(window=period_d).mean()
    return k_percent_smooth, d_percent

# Function to calculate ATR (Average True Range)
def calculate_atr(data, period):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    atr = true_range.rolling(window=period).mean()
    return atr

# Function to calculate ADX (Average Directional Index)
def calculate_adx(df, period):
    # Store the original index
    original_index = df.index
    # Reset the index to ensure it starts from 0
    df = df.reset_index(drop=True)
    # Calculate High, Low, and Close price differences
    df['H-PH'] = df['High'] - df['High'].shift(1)
    df['PL-L'] = df['Low'].shift(1) - df['Low']
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))

    # Calculate +DM and -DM
    df['+DM'] = np.where((df['H-PH'] > df['PL-L']) & (df['H-PH'] > 0), df['H-PH'], 0)
    df['-DM'] = np.where((df['PL-L'] > df['H-PH']) & (df['PL-L'] > 0), df['PL-L'], 0)

    # Initialize smoothed values
    df['Smooth +DM'] = np.nan
    df['Smooth -DM'] = np.nan
    df['Smooth TR'] = np.nan

    # Welles Wilder Smoothing for the first `period` periods
    df.loc[period-1, 'Smooth +DM'] = df['+DM'].iloc[:period].sum()
    df.loc[period-1, 'Smooth -DM'] = df['-DM'].iloc[:period].sum()
    df.loc[period-1, 'Smooth TR'] = df['TR'].iloc[:period].sum()

    # Apply Welles Wilder Smoothing for the rest of the periods
    for i in range(period, len(df)):
        df.loc[i, 'Smooth +DM'] = (df.loc[i-1, 'Smooth +DM'] * (period - 1) + df.loc[i, '+DM']) / period
        df.loc[i, 'Smooth -DM'] = (df.loc[i-1, 'Smooth -DM'] * (period - 1) + df.loc[i, '-DM']) / period
        df.loc[i, 'Smooth TR'] = (df.loc[i-1, 'Smooth TR'] * (period - 1) + df.loc[i, 'TR']) / period

    # Calculate +DI and -DI
    df['+DI'] = (df['Smooth +DM'] / df['Smooth TR']) * 100
    df['-DI'] = (df['Smooth -DM'] / df['Smooth TR']) * 100

    # Calculate DX
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100

    # Initialize ADX column
    df['ADX'] = np.nan
    df.loc[2 * period - 1, 'ADX'] = df['DX'].iloc[period-1:2*period-1].mean()  # First ADX value is the average of the first `period` DX values

    # Calculate ADX for the rest of the periods using Welles Wilder Smoothing
    for i in range(2 * period, len(df)):
        df.loc[i, 'ADX'] = (df.loc[i-1, 'ADX'] * (period - 1) + df.loc[i, 'DX']) / period

    # Restore the original index
    df.index = original_index

    return df['ADX']

# Function to create the SCRIP DataFrame for Bounce Trade
def create_scrip_dataframe(data, ema_periods, stochastic_periods, adx_period, atr_period):
    # Calculate EMAs, Stochastic Oscillator, ATR, and ADX
    data['EMA_8'] = calculate_ema(data, ema_periods['fast'])
    data['EMA_21'] = calculate_ema(data, ema_periods['mid1'])
    data['EMA_34'] = calculate_ema(data, ema_periods['mid2'])
    data['EMA_55'] = calculate_ema(data, ema_periods['mid3'])
    data['EMA_89'] = calculate_ema(data, ema_periods['slow'])

    data['%K'], data['%D'] = calculate_stochastic(data, stochastic_periods['k'], stochastic_periods['d'], stochastic_periods['smooth'])
    data['ATR'] = calculate_atr(data, atr_period)
    data['ADX'] = calculate_adx(data, adx_period)

    # Initialize 'Long/Short', 'Signal Number', and 'Stop-Loss' columns
    data['Long/Short'] = np.nan
    data['Signal Number'] = 0
    data['Stop-Loss'] = np.nan
    signal_number = 1
    active_trade = False
    entry_price = 0
    atr_multiple = 0

    # Loop through each row and identify Long/Short signals
    for i in range(4, len(data)):  # Start from 4th index to access previous days
        # Long Setup
        if (data['EMA_8'].iloc[i] > data['EMA_21'].iloc[i] > data['EMA_34'].iloc[i] >
            data['EMA_55'].iloc[i] > data['EMA_89'].iloc[i] and
            data['%K'].iloc[i] < 40 and data['ADX'].iloc[i] >= 20 and
            data['Close'].iloc[i] >= (data['EMA_21'].iloc[i] - data['ATR'].iloc[i]) and
            data['Close'].iloc[i] <= (data['EMA_21'].iloc[i] + data['ATR'].iloc[i]) and
            # 'Close' of entry date has to be greater than 'High' of the lowest 'Close' of the preceding 4 days
            data['Close'].iloc[i] > data['High'].iloc[data['Close'].iloc[i-4:i].argmin()]): #Entry Condition
            #data['Close'].iloc[i] > data['Low'].iloc[i-4:i].min()):  # Entry condition

            if not active_trade:
                # New Long Signal
                data.at[data.index[i], 'Long/Short'] = 'Long'
                data.at[data.index[i], 'Signal Number'] = signal_number
                entry_price = data['Close'].iloc[i]
                data.at[data.index[i], 'Stop-Loss'] = entry_price - data['ATR'].iloc[i]  # Initial stop-loss
                atr_multiple = 1  # Track multiples of ATR
                signal_number += 1
                active_trade = True

        # Short Setup
        elif (data['EMA_8'].iloc[i] < data['EMA_21'].iloc[i] < data['EMA_34'].iloc[i] <
              data['EMA_55'].iloc[i] < data['EMA_89'].iloc[i] and
              data['%K'].iloc[i] > 60 and data['ADX'].iloc[i] >= 20 and
              data['Close'].iloc[i] >= (data['EMA_21'].iloc[i] - data['ATR'].iloc[i]) and
              data['Close'].iloc[i] <= (data['EMA_21'].iloc[i] + data['ATR'].iloc[i]) and
              # 'Close' of the entry date has to be lower than the 'Low' of the highest 'Close' of the preceding 4 days
              data['Close'].iloc[i] < data['Low'].iloc[data['Close'].iloc[i-4:i].argmax()]): #Entry Condition
              #data['Close'].iloc[i] < data['High'].iloc[i-4:i].max()):  # Entry condition

            if not active_trade:
                # New Short Signal
                data.at[data.index[i], 'Long/Short'] = 'Short'
                data.at[data.index[i], 'Signal Number'] = signal_number
                entry_price = data['Close'].iloc[i]
                data.at[data.index[i], 'Stop-Loss'] = entry_price + data['ATR'].iloc[i]  # Initial stop-loss
                atr_multiple = 1  # Track multiples of ATR
                signal_number += 1
                active_trade = True

        # If there is an active trade, adjust the stop-loss based on the ATR multiples achieved
        if active_trade:
            if data['Long/Short'].iloc[i] == 'Long':
                # Adjust stop-loss for Long trades
                if data['Close'].iloc[i] >= entry_price + atr_multiple * data['ATR'].iloc[i]:
                    if atr_multiple == 1:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price  # Move SL to entry price
                    elif atr_multiple == 2:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price + 1 * data['ATR'].iloc[i]
                    elif atr_multiple == 3:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price + 2 * data['ATR'].iloc[i]
                    elif atr_multiple == 4:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price + 2.5 * data['ATR'].iloc[i]
                    elif atr_multiple == 5:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price + 3 * data['ATR'].iloc[i]
                    else:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price + (atr_multiple - 2) * data['ATR'].iloc[i]
                    atr_multiple += 1

            elif data['Long/Short'].iloc[i] == 'Short':
                # Adjust stop-loss for Short trades
                if data['Close'].iloc[i] <= entry_price - atr_multiple * data['ATR'].iloc[i]:
                    if atr_multiple == 1:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price  # Move SL to entry price
                    elif atr_multiple == 2:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price - 1 * data['ATR'].iloc[i]
                    elif atr_multiple == 3:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price - 2 * data['ATR'].iloc[i]
                    elif atr_multiple == 4:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price - 2.5 * data['ATR'].iloc[i]
                    elif atr_multiple == 5:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price - 3 * data['ATR'].iloc[i]
                    else:
                        data.at[data.index[i], 'Stop-Loss'] = entry_price - (atr_multiple - 2) * data['ATR'].iloc[i]
                    atr_multiple += 1

            # Stop-loss hit: Close the trade
            if data['Close'].iloc[i] < data['Stop-Loss'].iloc[i] and data['Long/Short'].iloc[i] == 'Long':
                active_trade = False
            elif data['Close'].iloc[i] > data['Stop-Loss'].iloc[i] and data['Long/Short'].iloc[i] == 'Short':
                active_trade = False

    return data
