import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load TSLA data from yfinance
tsla_data = yf.download("TSLA", start="2015-01-01", end="2023-04-23")

def visualize_stock_price_history():
    plt.figure(figsize=(16,8))
    plt.plot(tsla_data.index, tsla_data['Close'])
    plt.xlabel('Date')
    plt.ylabel('Closing price ($)')
    plt.title('Tesla (TSLA) Stock Price History')
    st.pyplot()

def visualize_technical_indicators():
    # Relative Strength Index (RSI)
    delta = tsla_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().abs()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # Stochastic Oscillator
    high_14, low_14 = tsla_data['High'].rolling(window=14).max(), tsla_data['Low'].rolling(window=14).min()
    k_percent = 100 * ((tsla_data['Close'] - low_14) / (high_14 - low_14))
    d_percent = k_percent.rolling(window=3).mean()
    
    # Plot RSI and Stochastic Oscillator
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,8))
    axes[0].plot(tsla_data.index, rsi)
    axes[0].set_title('Relative Strength Index (RSI)')
    axes[1].plot(tsla_data.index, k_percent, label='%K')
    axes[1].plot(tsla_data.index, d_percent, label='%D')
    axes[1].set_title('Stochastic Oscillator')
    axes[1].legend()
    st.pyplot()

def predict_stock_price(p, d, q):
    model = ARIMA(tsla_data['Close'], order=(p, d, q))
    results = model.fit()
    future_periods = 30
    forecast = results.forecast(steps=future_periods)

    # Plot forecasted stock prices
    plt.figure(figsize=(16,8))
    plt.plot(tsla_data.index, tsla_data['Close'])
    date_range = pd.date_range(start=tsla_data.index[-1], periods=future_periods+1, freq='D')[1:]
    plt.plot(date_range, forecast, label="Predicted Price")
    plt.title("Predicted Stock Prices for Next " + str(future_periods) + " Days")
    plt.xlabel("Date")
    plt.ylabel("Closing price ($)")
    plt.legend()
    st.pyplot()

# Streamlit app
def main():
    st.title("Tesla (TSLA) Stock Price Analysis")

    # Sidebar options
    option = st.sidebar.selectbox("Select an option", ("Stock Price History", "Technical Indicators", "Predict Future Prices"))

    if option == "Stock Price History":
        visualize_stock_price_history()

    elif option == "Technical Indicators":
        visualize_technical_indicators()

    else:
        st.write("Enter ARIMA model parameters:")
        p = st.slider("AR parameter (p)", 0, 5, 2)
        d = st.slider("Integration order (d)", 0, 5, 1)
        q = st.slider("MA parameter (q)", 0, 5, 2)

        predict_stock_price(p, d, q)

if __name__ == "__main__":
    main()
