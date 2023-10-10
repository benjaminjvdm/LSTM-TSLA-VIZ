# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load TSLA data from yfinance and preprocess data
tsla_data = yf.download("TSLA", start="2015-01-01")

# Global date range filter
start_date = st.sidebar.date_input("Start date", value=tsla_data.index.min())
end_date = st.sidebar.date_input("End date", value=tsla_data.index.max())

# Preprocess data
data = tsla_data.filter(['Close'])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Define a placeholder for plotting the chart
chart_placeholder = st.empty()

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

    # Display the chart
    chart_placeholder.pyplot(fig)

# ... Rest of the code is unchanged ...

# Main function
def main():
    st.sidebar.title("Tesla (TSLA) Stock Price Analysis")
    options = ["Stock Price History", "Stock Price Prediction"]
    choice = st.sidebar.selectbox("Select analysis type:", options)

    if choice == "Stock Price History":
        visualize_stock_price_history()
    elif choice == "Stock Price Prediction":
        build_and_train_model()

if __name__ == '__main__':
    main()
