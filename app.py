import streamlit as st
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima, ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load TSLA data from yfinance
tsla_data = yf.download("TSLA", start="2015-01-01")

# Global date range filter
start_date = st.sidebar.date_input("Start date", value=tsla_data.index.min())
end_date = st.sidebar.date_input("End date", value=tsla_data.index.max())

def visualize_stock_price_history():
    # Filter data based on selected dates
    tsla_data_filtered = tsla_data.loc[start_date:end_date]

    # Relative Strength Index (RSI)
    delta = tsla_data_filtered['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().abs()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # Stochastic Oscillator
    high_14, low_14 = tsla_data_filtered['High'].rolling(window=14).max(), tsla_data_filtered['Low'].rolling(window=14).min()
    k_percent = 100 * ((tsla_data_filtered['Close'] - low_14) / (high_14 - low_14))
    d_percent = k_percent.rolling(window=3).mean()

    # Plot stock price history, RSI, and Stochastic Oscillator
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16,12), sharex=True)
    axes[0].plot(tsla_data_filtered.index, tsla_data_filtered['Close'])
    axes[0].set_title('Tesla (TSLA) Stock Price History')
    axes[0].set_ylabel('Closing price ($)')
    axes[1].plot(tsla_data_filtered.index, rsi)
    axes[1].set_title('Relative Strength Index (RSI)')
    axes[2].plot(tsla_data_filtered.index, k_percent, label='%K')
    axes[2].plot(tsla_data_filtered.index, d_percent, label='%D')
    axes[2].set_title('Stochastic Oscillator')
    axes[2].legend()
    st.pyplot(fig)

def predict_stock_price(use_auto_arima):
    # Filter data based on selected dates
    tsla_data_filtered = tsla_data.loc[start_date:end_date]

    if use_auto_arima:
        # Perform auto-ARIMA to find optimal parameters
        model = auto_arima(tsla_data_filtered['Close'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        p, d, q = model.order
        st.write(f"Best ARIMA parameters found by auto-ARIMA: (p = {p}, d = {d}, q = {q})")
    
    else:
        st.write("Enter ARIMA model parameters:")
        p = st.slider("AR parameter (p)", 0, 10, 2)
        d = st.slider("Integration order (d)", 0, 10, 1)
        q = st.slider("MA parameter (q)", 0, 10, 2)
    
    # Build the ARIMA model with selected or best parameters
    model = ARIMA(tsla_data_filtered['Close'], order=(p, d, q))
    results = model.fit()
    future_periods = 30
    forecast = results.forecast(steps=future_periods)

    # Plot forecasted stock prices
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(tsla_data_filtered.index, tsla_data_filtered['Close'])
    date_range = pd.date_range(start=tsla_data_filtered.index[-1], periods=future_periods+1, freq='D')[1:]
    ax.plot(date_range, forecast, label="Predicted Price")
    ax.set_title("Predicted Stock Prices for Next " + str(future_periods) + " Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing price ($)")
    ax.legend()
    st.pyplot

#Streamlit
def main():
st.title("Tesla (TSLA) Stock Price Analysis")

# Sidebar options
option = st.sidebar.selectbox("Select an option", ("Stock Price History and Technical Indicators", "Predict Future Prices"))

if option == "Stock Price History and Technical Indicators":
    visualize_stock_price_history()

else:
    use_auto_arima = st.checkbox("Use auto-ARIMA to find optimal parameters", value=True)
    predict_stock_price(use_auto_arima)

if name == "main":
main()
