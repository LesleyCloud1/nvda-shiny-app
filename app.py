#shiny run --reload app.py
from shiny import App, render, ui
import pandas as pd
import shiny
import yfinance as yf
import finnhub
from datetime import datetime, timedelta
import shinyswatch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


API_KEY = "d4mdr39r01qjidhv7qa0d4mdr39r01qjidhv7qag"
finnhub_client = finnhub.Client(api_key=API_KEY)


def get_current_date_last_seven():
    current_date = datetime.now()
    date_7_days_ago = current_date - timedelta(days=7)
    cd = current_date.strftime("%Y-%m-%d")
    lsd = date_7_days_ago.strftime("%Y-%m-%d")
    print("Current date:", cd)
    print("Date 7 days ago:", lsd)
    return (cd, lsd)


def ts_to_date_str(timestamp):
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime("%m-%d-%Y")
    return str(formatted_date)


def get_news():
    symbol = "NVDA"
    d = get_current_date_last_seven()
    begin = d[1]
    end = d[0]
    news = finnhub_client.company_news(symbol, _from=begin, to=end)
    print("Number of news stories returned: " + str(len(news)))
    return news


news = get_news()


def get_dataframe():
    ticker = "NVDA"
    start_date = "2000-01-01"
    end_date = "2050-12-31"
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = ["_".join(col) for col in df.columns]
    df["Date"] = df.index
    df = df.reset_index(drop=True)

    column_names = ["Date", "Close", "High", "Low", "Open", "Volume"]
    rename_map = {
        "Date": "Date",
        "Close_NVDA": "Close",
        "High_NVDA": "High",
        "Low_NVDA": "Low",
        "Open_NVDA": "Open",
        "Volume_NVDA": "Volume",
    }
    df = df.rename(columns=rename_map)[column_names]

    columns_to_round = {"Close": 3, "High": 3, "Low": 3, "Open": 3}
    df = df.round(columns_to_round)

    df = df.sort_values(by="Date", ascending=True)
    df["Date"] = df["Date"].dt.date

    print(
        "{x} rows of data returned for data between dates {start_date} and {end_date}".format(
            x=len(df), start_date=start_date, end_date=end_date
        )
    )
    return df


df = get_dataframe()


# ========== LINEAR REGRESSION FUNCTIONS ==========
def predict_next_close(dataframe: pd.DataFrame) -> float:
    data = dataframe.copy().sort_values("Date")
    data["Close_lag1"] = data["Close"].shift(1)
    data["Close_lag2"] = data["Close"].shift(2)
    data = data.dropna(subset=["Close", "Close_lag1", "Close_lag2"])

    if len(data) < 3:
        return float("nan")

    X = data[["Close_lag1", "Close_lag2"]].values
    y = data["Close"].values

    model = LinearRegression()
    model.fit(X, y)

    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    X_next = np.array([[last_row["Close"], prev_row["Close"]]])
    next_close = float(model.predict(X_next)[0])
    return round(next_close, 3)


def predict_next_high(dataframe: pd.DataFrame) -> float:
    data = dataframe.copy().sort_values("Date")
    data["High_lag1"] = data["High"].shift(1)
    data["High_lag2"] = data["High"].shift(2)
    data["Close_lag1"] = data["Close"].shift(1)

    data = data.dropna(subset=["High", "High_lag1", "High_lag2", "Close_lag1"])

    if len(data) < 3:
        return float("nan")

    X = data[["High_lag1", "High_lag2", "Close_lag1"]].values
    y = data["High"].values

    model = LinearRegression()
    model.fit(X, y)

    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    X_next = np.array([[last_row["High"], prev_row["High"], last_row["Close"]]])
    next_high = float(model.predict(X_next)[0])
    return round(next_high, 3)


def predict_next_low(dataframe: pd.DataFrame) -> float:
    data = dataframe.copy().sort_values("Date")
    data["Low_lag1"] = data["Low"].shift(1)
    data["Low_lag2"] = data["Low"].shift(2)
    data["Close_lag1"] = data["Close"].shift(1)

    data = data.dropna(subset=["Low", "Low_lag1", "Low_lag2", "Close_lag1"])

    if len(data) < 3:
        return float("nan")

    X = data[["Low_lag1", "Low_lag2", "Close_lag1"]].values
    y = data["Low"].values

    model = LinearRegression()
    model.fit(X, y)

    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    X_next = np.array([[last_row["Low"], prev_row["Low"], last_row["Close"]]])
    next_low = float(model.predict(X_next)[0])
    return round(next_low, 3)


def backtest_close_model(dataframe: pd.DataFrame, lookback_days: int = 250):
    data = dataframe.copy().sort_values("Date").reset_index(drop=True)

    preds = []
    actuals = []

    start_idx = max(3, len(data) - lookback_days)

    for i in range(start_idx, len(data)):
        train = data.iloc[:i].copy()

        train["Close_lag1"] = train["Close"].shift(1)
        train["Close_lag2"] = train["Close"].shift(2)
        train = train.dropna(subset=["Close", "Close_lag1", "Close_lag2"])

        if len(train) < 3:
            continue

        X_train = train[["Close_lag1", "Close_lag2"]].values
        y_train = train["Close"].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        row = data.iloc[i]
        lag1 = data.iloc[i - 1]["Close"]
        lag2 = data.iloc[i - 2]["Close"]
        X_test = np.array([[lag1, lag2]])
        y_pred = float(model.predict(X_test)[0])
        y_true = row["Close"]

        preds.append(y_pred)
        actuals.append(y_true)

    if len(actuals) == 0:
        return float("nan"), float("nan")

    mae = mean_absolute_error(actuals, preds)
    mse = mean_squared_error(actuals, preds)
    rmse = mse ** 0.5

    return mae, rmse


# ========== LSTM FUNCTIONS ==========
def create_lstm_dataset(data, lookback=60):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(lookback=60):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_next_close_lstm(dataframe: pd.DataFrame, lookback=60) -> float:
    """Predict next close price using LSTM"""
    try:
        print("Training LSTM for next-day prediction...")
        data = dataframe.copy().sort_values("Date")
        close_prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        # Create training data
        X_train, y_train = create_lstm_dataset(scaled_data, lookback)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Build and train model
        model = build_lstm_model(lookback)
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
        
        # Predict next day
        last_sequence = scaled_data[-lookback:]
        last_sequence = last_sequence.reshape(1, lookback, 1)
        
        prediction_scaled = model.predict(last_sequence, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        print("LSTM next-day prediction complete!")
        return round(float(prediction[0][0]), 3)
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return float("nan")


def backtest_lstm_model(dataframe: pd.DataFrame, lookback=60, test_days=10):
    """Backtest LSTM model on recent data"""
    try:
        print(f"Starting LSTM backtest on last {test_days} days...")
        data = dataframe.copy().sort_values("Date").reset_index(drop=True)
        close_prices = data['Close'].values.reshape(-1, 1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        preds = []
        actuals = []
        
        # Start from enough data for training
        start_idx = max(lookback + 100, len(data) - test_days)
        
        for idx, i in enumerate(range(start_idx, len(data))):
            print(f"  Processing day {idx+1}/{test_days}...")
            # Use all data up to day i for training
            train_data = scaled_data[:i]
            
            if len(train_data) < lookback + 10:
                continue
                
            X_train, y_train = create_lstm_dataset(train_data, lookback)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            # Build and train model
            model = build_lstm_model(lookback)
            model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)
            
            # Predict day i
            last_sequence = train_data[-lookback:]
            last_sequence = last_sequence.reshape(1, lookback, 1)
            
            pred_scaled = model.predict(last_sequence, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)
            
            preds.append(float(pred[0][0]))
            actuals.append(data.iloc[i]['Close'])
        
        if len(actuals) == 0:
            return float("nan"), float("nan")
        
        mae = mean_absolute_error(actuals, preds)
        mse = mean_squared_error(actuals, preds)
        rmse = mse ** 0.5
        
        print("LSTM backtest complete!")
        return mae, rmse
    except Exception as e:
        print(f"LSTM backtest error: {e}")
        return float("nan"), float("nan")


# ========== CALCULATE ALL PREDICTIONS ==========
print("\n" + "="*50)
print("Calculating Linear Regression predictions...")
print("="*50)
predicted_next_close_lr = predict_next_close(df)
predicted_next_high_lr = predict_next_high(df)
predicted_next_low_lr = predict_next_low(df)
mae_close_lr, rmse_close_lr = backtest_close_model(df, lookback_days=250)

print("\n" + "="*50)
print("Calculating LSTM predictions...")
print("="*50)
predicted_next_close_lstm = predict_next_close_lstm(df, lookback=60)
mae_close_lstm, rmse_close_lstm = backtest_lstm_model(df, lookback=60, test_days=10)

print("\n" + "="*50)
print("=== LINEAR REGRESSION RESULTS ===")
print("="*50)
print("PREDICTED_NEXT_CLOSE:", predicted_next_close_lr)
print("PREDICTED_NEXT_HIGH:", predicted_next_high_lr)
print("PREDICTED_NEXT_LOW:", predicted_next_low_lr)
print("BACKTEST MAE:", mae_close_lr)
print("BACKTEST RMSE:", rmse_close_lr)

print("\n" + "="*50)
print("=== LSTM RESULTS ===")
print("="*50)
print("PREDICTED_NEXT_CLOSE:", predicted_next_close_lstm)
print("BACKTEST MAE:", mae_close_lstm)
print("BACKTEST RMSE:", rmse_close_lstm)
print("="*50 + "\n")


# ========== UI ==========
app_ui = ui.page_fluid(
    ui.row(
        ui.h1(
            "Data Show and Tell: Exploring NVIDIA Finance Data with Python Shiny",
            style="text-align: center; margin-top: 20px;",
        )
    ),
    ui.row(
        ui.h3(
            "NVDA Data Historical Prices",
            style="text-align: center; margin-top: 20px;",
        ),
        ui.card(
            ui.output_data_frame("my_table"),
            ui.download_button("download", "Download CSV"),
            height="400px",
        ),
        ui.h3(
            "NVDA Stock Close Prices Over Time",
            style="text-align: center; margin-top: 20px;",
        ),
        ui.output_plot("my_plot"),
        ui.h3(
            "NVDA Next-Day Price Predictions - Model Comparison",
            style="text-align: center; margin-top: 20px;",
        ),
        ui.card(
            ui.output_text("next_price_text"),
            style="text-align: center; font-size: 16px; padding: 10px;",
        ),
        ui.h3(
            "NVDA Data Historical Stock Splits",
            style="text-align: center; margin-top: 20px;",
            class_="col-md-12",
        ),
        ui.card(ui.output_data_frame("stock_splits")),
    ),
    ui.div(
        ui.tags.style(
            """
            .center-text {
                text-align: center;
            }
        """
        ),
        ui.h3("NVDA News Feed (Last 7 days)"),
        ui.div(
            ui.div(
                *[
                    ui.div(
                        ui.card(
                            ui.card_header(card["headline"]),
                            ui.card_body("Date: " + ts_to_date_str(card["datetime"])),
                            ui.card_body("Source: " + card["source"]),
                            ui.tags.a(
                                "Read More",
                                href=card["url"],
                                target="_blank",
                            ),
                            class_="col-md-12",
                        ),
                        class_="card-container",
                    )
                    for card in news
                ],
                class_="scrollable-container",
            ),
            class_="outer-container",
        ),
        class_="center-text",
    ),
    ui.tags.style(
        """
        .outer-container {
            max-height: 400px;
            width: 100%;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .scrollable-container {
            max-height: 400px;
            width: 100%;
            overflow-y: auto;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            background-color: #fff;
        }
        .card-container {
            width:100%;
        }
        """
    ),
    theme=shinyswatch.theme.darkly,
)


# ========== SERVER ==========
def server(input, output, session):
    @output
    @render.data_frame
    def my_table():
        display_df = df.copy()
        display_df = display_df.sort_values(by="Date", ascending=False)
        display_df["Volume"] = display_df["Volume"].apply(lambda x: f"{int(x):,}")
        return display_df

    @output
    @render.data_frame
    def stock_splits():
        ticker = "NVDA"
        stock_splits = yf.Ticker(ticker).splits
        df_splits = pd.DataFrame(stock_splits)
        df_splits["Date"] = df_splits.index
        df_splits = df_splits.reset_index(drop=True)
        df_splits["Date"] = df_splits["Date"].apply(lambda x: x.date())
        df_splits = df_splits.rename(columns={"Stock Splits": "Multiplier"})
        df_splits = df_splits[["Date", "Multiplier"]]
        return df_splits

    @session.download(filename="NVDA_Stock_Data.csv")
    def download():
        yield df.to_csv(index=False)

    @output
    @render.text
    def next_price_text():
        last_date = df["Date"].max()
        row = df.loc[df["Date"] == last_date].iloc[0]
        last_close = row["Close"]
        last_high = row["High"]
        last_low = row["Low"]

        msg = f"📅 Last close on {last_date}: ${last_close:.3f} | High: ${last_high:.3f} | Low: ${last_low:.3f}\n\n"
        
        msg += "🔵 LINEAR REGRESSION MODEL:\n"
        if not np.isnan(predicted_next_close_lr):
            msg += f"   • Predicted next close: ${predicted_next_close_lr:.3f}\n"
            msg += f"   • Predicted next high: ${predicted_next_high_lr:.3f}\n"
            msg += f"   • Predicted next low: ${predicted_next_low_lr:.3f}\n"
        else:
            msg += "   • N/A\n"
        
        if not np.isnan(mae_close_lr):
            msg += f"   • Backtest (250 days) - MAE: ${mae_close_lr:.3f}, RMSE: ${rmse_close_lr:.3f}\n\n"
        
        msg += "🟢 LSTM MODEL (Deep Learning):\n"
        if not np.isnan(predicted_next_close_lstm):
            msg += f"   • Predicted next close: ${predicted_next_close_lstm:.3f}\n"
        else:
            msg += "   • N/A\n"
            
        if not np.isnan(mae_close_lstm):
            msg += f"   • Backtest (10 days) - MAE: ${mae_close_lstm:.3f}, RMSE: ${rmse_close_lstm:.3f}\n"
        else:
            msg += "   • Backtest in progress...\n"

        return msg

    @shiny.render.plot
    def my_plot():
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df["Date"], df["Close"], label="Historical Close Price", color="blue", linewidth=2)

        last_date = df["Date"].max()
        next_date = last_date + timedelta(days=1)

        # Linear Regression predictions
        if not np.isnan(predicted_next_close_lr):
            ax.scatter([next_date], [predicted_next_close_lr], color="red", s=100, 
                      label="LR: Predicted Close", marker='o', zorder=5)
            ax.scatter([next_date], [predicted_next_high_lr], color="orange", s=100,
                      label="LR: Predicted High", marker='^', zorder=5)
            ax.scatter([next_date], [predicted_next_low_lr], color="green", s=100,
                      label="LR: Predicted Low", marker='v', zorder=5)

        # LSTM prediction
        if not np.isnan(predicted_next_close_lstm):
            ax.scatter([next_date], [predicted_next_close_lstm], color="purple", s=150,
                      label="LSTM: Predicted Close", marker='*', zorder=6)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price (USD)", fontsize=12)
        ax.set_title("NVDA Stock Price: Historical + Predictions", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig


app = shiny.App(app_ui, server)
