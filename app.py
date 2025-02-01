import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import requests

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load TSLA data from yfinance
@st.cache_data
def load_data():
    return yf.download("TSLA", start="2015-01-01")

tsla_data = load_data()

# Sidebar for date range and presets
st.sidebar.title("Options")
start_date = st.sidebar.date_input("Start date", value=tsla_data.index.min())
end_date = st.sidebar.date_input("End date", value=tsla_data.index.max())

preset_options = ["6M", "1Y", "5Y", "Max"]
selected_preset = st.sidebar.selectbox("Or Select a Preset Date Range:", preset_options)

if selected_preset == "6M":
    start_date = end_date - pd.DateOffset(months=6)
elif selected_preset == "1Y":
    start_date = end_date - pd.DateOffset(years=1)
elif selected_preset == "5Y":
    start_date = end_date - pd.DateOffset(years=5)
elif selected_preset == "Max":
    start_date = tsla_data.index.min()

# Filter data based on selected dates
tsla_data_filtered = tsla_data.loc[start_date:end_date]

# Visualize stock price history
def visualize_stock_price_history(data):
    st.subheader("Tesla (TSLA) Stock Price History")

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().abs()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # Stochastic Oscillator
    high_14, low_14 = data['High'].rolling(window=14).max(), data['Low'].rolling(window=14).min()
    k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    d_percent = k_percent.rolling(window=3).mean()

    # Add checkboxes for the indicators
    show_rsi = st.sidebar.checkbox('Show RSI', value=True)
    show_stochastic_oscillator = st.sidebar.checkbox('Show Stochastic Oscillator', value=True)

    # Plot stock price history, RSI, and Stochastic Oscillator
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 12), sharex=True)
    axes[0].plot(data.index, data['Close'])
    axes[0].set_title('Tesla (TSLA) Stock Price History')
    axes[0].set_ylabel('Closing price ($)')

    if show_rsi:
        axes[1].plot(data.index, rsi)
        axes[1].set_title('Relative Strength Index (RSI)')

    if show_stochastic_oscillator:
        axes[2].plot(data.index, k_percent, label='%K')
        axes[2].plot(data.index, d_percent, label='%D')
        axes[2].set_title('Stochastic Oscillator')
        axes[2].legend()

    st.pyplot(fig)

# Build and train the LSTM model
def build_and_train_model(data):
    st.subheader("Stock Price Prediction - LSTM")

    # Preprocess data
    dataset = data.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training dataset
    training_data_len = int(len(dataset) * 0.8)
    train_data = scaled_data[:training_data_len, :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Test dataset
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Evaluate the model
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    st.write('Root Mean Squared Error:', rmse)

    # Plot predictions
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(train['Close'])
    ax.plot(valid[['Close', 'Predictions']])
    ax.legend(['Train', 'Validation', 'Prediction'], loc='upper left')
    ax.set_title('Tesla (TSLA) Stock Price Prediction - LSTM')
    ax.set_ylabel('Closing price ($)')
    st.pyplot(fig)

def main():
    st.title("Tesla (TSLA) Stock Price Analysis")

    options = ["Stock Indicator Analysis", "All Models", "Stock Price Prediction - LSTM", "Stock Price Prediction - SVM", "Stock Price Prediction - LightGBM"]
    choice = st.sidebar.selectbox("Select analysis type:", options)

    if choice == "Stock Indicator Analysis":
        visualize_stock_price_history(tsla_data_filtered)
    elif choice == "All Models":
        build_and_train_model(tsla_data_filtered)
        # Add SVM and LightGBM functions here
    elif choice == "Stock Price Prediction - LSTM":
        build_and_train_model(tsla_data_filtered)

    # Footer
    with st.sidebar:
        st.subheader("About the Author")
        image_url = "https://avatars.githubusercontent.com/u/97449931?v=4"
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            st.image(response.content, caption="Moon Benjee (문벤지)")
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading image: {e}")

        st.markdown(
            """
            This app was Built with ❤️ by **Benjee(문벤지)**. 
            You can connect with me on: [LinkedIn](https://www.linkedin.com/in/benjaminjvdm/)
            """
        )

if __name__ == '__main__':
    main()
