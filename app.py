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
    st.pyplot()

def predict_stock_price(p, d, q, use_grid_search):
    # Filter data based on selected dates
    tsla_data_filtered = tsla_data.loc[start_date:end_date]

    if use_grid_search:
        # Perform grid search to find optimal parameters
        p_values = range(0, 3)
        d_values = range(0, 3)
        q_values = range(0, 3)
        best_aic, best_order = np.inf, None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(tsla_data_filtered['Close'], order=(p, d, q))
                        results = model.fit()
                        aic = results.aic
                        if aic < best_aic:
                            best_aic, best_order = aic, (p, d, q)
                    except:
                        continue

        p, d, q = best_order
        st.write(f"Best ARIMA parameters found by grid search: (p = {p}, d = {d}, q = {q})")
    
    # Build the ARIMA model with selected or best parameters
    model = ARIMA(tsla_data_filtered['Close'], order=(p, d, q))
    results = model.fit()
    future_periods = 30
    forecast = results.forecast(steps=future_periods)

    # Plot forecasted stock prices
    plt.figure(figsize=(16,8))
    plt.plot(tsla_data_filtered.index, tsla_data_filtered['Close'])
    date_range = pd.date_range(start=tsla_data_filtered.index[-1], periods=future_periods+1, freq='D')[1:]
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
    option = st.sidebar.selectbox("Select an option", ("Stock Price History and Technical Indicators", "Predict Future Prices"))

    if option == "Stock Price History and Technical Indicators":
        visualize_stock_price_history()

    else:
        st.write("Enter ARIMA model parameters:")
        p = st.slider("AR parameter (p)", 0, 5, 2)
        d = st.slider("Integration order (d)", 0, 5, 1)
        q = st.slider("MA parameter (q)", 0, 5, 2)
        use_grid_search = st.checkbox("Use grid search to find optimal parameters", value=False)

        predict_stock_price(p, d, q, use_grid_search)

if __name__ == "__main__":
    main()
