# Tesla (TSLA) Stock Price Analysis

This project analyzes the historical stock price of Tesla (TSLA) using the Python libraries `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `yfinance`, and `tensorflow`. 

## Project Structure

The project consists of a single file, `app.py`, which contains the following functions:

* `visualize_stock_price_history()` - This function visualizes the historical TSLA stock price along with the Relative Strength Index (RSI) and Stochastic Oscillator. It takes no input parameters.

* `build_and_train_model()` - This function builds and trains an LSTM model to predict future TSLA stock prices. It takes no input parameters.

* `main()` - This is the main function that runs the project. It presents the user with a sidebar that allows them to choose between visualizing the stock price history or predicting future stock prices. Depending on the user's choice, it calls either `visualize_stock_price_history()` or `build_and_train_model()`.

## Installation

To run this project, you need to have Python 3.x installed on your machine along with the required libraries mentioned above. To install these libraries, you can use pip as follows:

```
pip install streamlit pandas numpy matplotlib seaborn yfinance tensorflow
```

## Usage

To run the project, navigate to the directory containing `app.py` and run the following command in the terminal:

```
streamlit run app.py
```

This will start a local server and open the project in your web browser. From there, you can use the sidebar to select the analysis type you want to perform: stock price history or stock price prediction.

If you select "Stock Price History", the application will display a plot of the historical TSLA stock price along with RSI and Stochastic Oscillator. You can select the date range you want to visualize using the date inputs on the sidebar.

If you select "Stock Price Prediction", the application will build and train an LSTM model to predict future TSLA stock prices based on historical data. The predicted stock price and actual stock price will be plotted for comparison.
