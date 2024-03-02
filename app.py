import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import mplfinance as mpf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load TSLA data from yfinance and preprocess data
tsla_data = yf.download("TSLA", start="2015-01-01")

# Global date range filter
start_date = st.sidebar.date_input("Start date", value=tsla_data.index.min())
end_date = st.sidebar.date_input("End date", value=tsla_data.index.max())

# Add presets for the date selector
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

# Preprocess data
data = tsla_data.filter(['Close'])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

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

    # Add checkboxes for the indicators
    show_rsi = st.sidebar.checkbox('Show RSI', value=True)
    show_stochastic_oscillator = st.sidebar.checkbox('Show Stochastic Oscillator', value=True)

    # Plot stock price history, RSI, and Stochastic Oscillator
    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(16,12), sharex=True)
    axes[0].plot(tsla_data_filtered.index, tsla_data_filtered['Close'])
    axes[0].set_title('Tesla (TSLA) Stock Price History')
    axes[0].set_ylabel('Closing price ($)')

    if show_rsi:
        axes[1].plot(tsla_data_filtered.index, rsi)
        axes[1].set_title('Relative Strength Index (RSI)')

    if show_stochastic_oscillator:
        axes[2].plot(tsla_data_filtered.index, k_percent, label='%K')
        axes[2].plot(tsla_data_filtered.index, d_percent, label='%D')
        axes[2].set_title('Stochastic Oscillator')
        axes[2].legend()
    
    st.pyplot()

# Build and train the LSTM model
def build_and_train_model():
    # Filter data based on selected dates
    tsla_data_filtered = tsla_data.loc[start_date:end_date]

    # Preprocess data
    data = tsla_data_filtered.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training dataset
    training_data_len = int(len(dataset) * 0.8)
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    # Ensure train_data has at least 60 elements
    if len(train_data) < 60:
        train_data = np.pad(train_data, ((60 - len(train_data), 0), (0, 0)), 'constant', constant_values=(0,))

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    # Compile the model
    opt = RMSprop(lr=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error')

    # Train the model
    epochs = 15
    batch_size = 64
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Test dataset
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Predict the next price
    next_price = model.predict(x_test[-1].reshape(1, -1, 1))
    next_price = scaler.inverse_transform(next_price)
    st.write('Next predicted price (LSTM): $', round(next_price[0][0], 2))

    # Evaluate the model
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    st.write('Root Mean Squared Error:', rmse)

    # Plot predictions vs actual data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid.loc[:, 'Predictions'] = predictions

    _, ax = plt.subplots(figsize=(16,8))
    ax.plot(train['Close'])
    ax.plot(valid[['Close', 'Predictions']])
    ax.legend(['Train', 'Validation', 'Prediction'], loc='upper left')
    ax.set_title('Tesla (TSLA) Stock Price Prediction - LTSM')
    ax.set_ylabel('Closing price ($)')
    st.pyplot()

# Build and train the SVM model
def build_and_train_svm():
    # Filter data based on selected dates
    tsla_data_filtered = tsla_data.loc[start_date:end_date]

    # Preprocess data
    data = tsla_data_filtered.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training and testing datasets
    training_data_len = int(len(dataset) * 0.8)
    x_train = scaled_data[:training_data_len]
    y_train = dataset[:training_data_len]
    x_test = scaled_data[training_data_len:]
    y_test = dataset[training_data_len:]

    # Build SVM model
    model = SVR(kernel='linear')
    model.fit(x_train, y_train)

    # Make predictions
    predictions = model.predict(x_test)

    # Predict the next price
    next_price = model.predict(x_test[-1].reshape(1, -1))
    st.write('Next predicted price (SVM): $', next_price[0])

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    st.write('Root Mean Squared Error:', rmse)

    # Plot predictions vs actual data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid.loc[:, 'Predictions'] = predictions

    _, ax = plt.subplots(figsize=(16,8))
    ax.plot(train['Close'])
    ax.plot(valid[['Close', 'Predictions']])
    ax.legend(['Train', 'Validation', 'Prediction'], loc='upper left')
    ax.set_title('Tesla (TSLA) Stock Price Prediction - SVM')
    ax.set_ylabel('Closing price ($)')
    st.pyplot()

# Build and train the LightGBM model
def build_and_train_lgbm():
    # Filter data based on selected dates
    tsla_data_filtered = tsla_data.loc[start_date:end_date]

    # Preprocess data
    data = tsla_data_filtered.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training and testing datasets
    training_data_len = int(len(dataset) * 0.8)
    x_train = scaled_data[:training_data_len]
    y_train = dataset[:training_data_len]
    x_test = scaled_data[training_data_len:]
    y_test = dataset[training_data_len:]

    # Build LightGBM model
    model = LGBMRegressor()
    model.fit(x_train, y_train)

    # Make predictions
    predictions = model.predict(x_test)

    # Predict the next price
    next_price = model.predict(x_test[-1].reshape(1, -1))
    st.write('Next predicted price (LightGBM): $', next_price[0])

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    st.write('Root Mean Squared Error:', rmse)

    # Plot predictions vs actual data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid.loc[:, 'Predictions'] = predictions

    _, ax = plt.subplots(figsize=(16,8))
    ax.plot(train['Close'])
    ax.plot(valid[['Close', 'Predictions']])
    ax.legend(['Train', 'Validation', 'Prediction'], loc='upper left')
    ax.set_title('Tesla (TSLA) Stock Price Prediction - LightGBM')
    ax.set_ylabel('Closing price ($)')
    st.pyplot()

def main():
    st.title("Tesla (TSLA) Stock Price Analysis")
    st.sidebar.title("Options")

    options = ["Stock Price History", "All Models", "Stock Price Prediction - LSTM", "Stock Price Prediction - SVM", "Stock Price Prediction - LightGBM"]
    choice = st.sidebar.selectbox("Select analysis type:", options)

    if choice == "Stock Price History":
        visualize_stock_price_history()
    elif choice == "All Models":
        build_and_train_model()
        build_and_train_svm()
        build_and_train_lgbm()
    elif choice == "Stock Price Prediction - LSTM":
        build_and_train_model()
    elif choice == "Stock Price Prediction - SVM":
        build_and_train_svm()
    elif choice == "Stock Price Prediction - LightGBM":
        build_and_train_lgbm()


    # Add a footer
    st.sidebar.markdown("---")
    st.sidebar.text("Built with ❤️ by Benjee(문벤지)")

if __name__ == '__main__':
    main()
