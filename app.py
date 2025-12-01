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
import numpy as np

API_KEY = "d4mdr39r01qjidhv7qa0d4mdr39r01qjidhv7qag"  # TODO: Replace with your Finnhub API key
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


news = get_news()  # Retrieve News with Finnhub API


def get_dataframe():
    ticker = "NVDA"
    start_date = "2000-01-01"  # IPO Date.
    end_date = "2050-12-31"  # Future Date :)
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = ["_".join(col) for col in df.columns]
    df["Date"] = df.index
    df = df.reset_index(drop=True)

    column_names = ["Date", "Close", "High", "Low", "Open", "Volume"]
    rename_map = {
        "Date": "Date",  # Map old column names to new names
        "Close_NVDA": "Close",
        "High_NVDA": "High",
        "Low_NVDA": "Low",
        "Open_NVDA": "Open",
        "Volume_NVDA": "Volume",
    }
    df = df.rename(columns=rename_map)[column_names]  # Rename and reorder columns

    # Keep numeric for modeling
    columns_to_round = {"Close": 3, "High": 3, "Low": 3, "Open": 3}
    df = df.round(columns_to_round)

    # Sort oldest -> newest for time series work
    df = df.sort_values(by="Date", ascending=True)
    df["Date"] = df["Date"].dt.date

    print(
        "{x} rows of data returned for data between dates {start_date} and {end_date}".format(
            x=len(df), start_date=start_date, end_date=end_date
        )
    )
    return df


df = get_dataframe()


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
    # Predict tomorrow's high using lagged highs and closes
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
    # Predict tomorrow's low using lagged lows and closes
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
    """
    Walk-forward backtest for close price:
    For each day in the last `lookback_days`, fit on all prior data,
    predict that day's close, and compare to actual.
    """
    data = dataframe.copy().sort_values("Date").reset_index(drop=True)

    preds = []
    actuals = []

    # start index so we always have at least 3 prior points for training
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

        # predict for day i using its lags
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

    # MAE directly
    mae = mean_absolute_error(actuals, preds)

    # RMSE manually (no squared= argument)
    mse = mean_squared_error(actuals, preds)
    rmse = mse ** 0.5

    return mae, rmse


# Global predictions + backtest metrics
predicted_next_close = predict_next_close(df)
predicted_next_high = predict_next_high(df)
predicted_next_low = predict_next_low(df)
mae_close, rmse_close = backtest_close_model(df, lookback_days=250)

print("PREDICTED_NEXT_CLOSE:", predicted_next_close)
print("PREDICTED_NEXT_HIGH:", predicted_next_high)
print("PREDICTED_NEXT_LOW:", predicted_next_low)
print("BACKTEST MAE (close):", mae_close)
print("BACKTEST RMSE (close):", rmse_close)


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
            "NVDA Next-Day Price Predictions",
            style="text-align: center; margin-top: 20px;",
        ),
        ui.card(
            ui.output_text("next_price_text"),
            style="text-align: center; font-size: 18px; padding: 10px;",
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
            grid-template-columns: repeat(3, 1fr); /* 3 cards per row */
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


def server(input, output, session):
    @output
    @render.data_frame
    def my_table():
        display_df = df.copy()
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

        msg = (
            f"Last close on {last_date}: ${last_close:.3f} | "
            f"Last high: ${last_high:.3f} | "
            f"Last low: ${last_low:.3f}"
        )
        if not np.isnan(predicted_next_close):
            msg += f" | Predicted next close: ${predicted_next_close:.3f}"
        else:
            msg += " | Predicted next close: N/A"

        if not np.isnan(predicted_next_high):
            msg += f" | Predicted next high: ${predicted_next_high:.3f}"
        else:
            msg += " | Predicted next high: N/A"

        if not np.isnan(predicted_next_low):
            msg += f" | Predicted next low: ${predicted_next_low:.3f}"
        else:
            msg += " | Predicted next low: N/A"

        if not np.isnan(mae_close) and not np.isnan(rmse_close):
            msg += (
                f" || Backtest (last 250 days) MAE: ${mae_close:.3f}, "
                f"RMSE: ${rmse_close:.3f}"
            )
        else:
            msg += " || Backtest metrics not available."

        return msg

    @shiny.render.plot
    def my_plot():
        fig, ax = plt.subplots()

        # Historical close prices
        ax.plot(df["Date"], df["Close"], label="Close Price", color="blue", linewidth=2)

        # Predicted next-day close, high, and low
        last_date = df["Date"].max()
        next_date = last_date + timedelta(days=1)

        if not np.isnan(predicted_next_close):
            ax.scatter(
                [next_date],
                [predicted_next_close],
                color="red",
                label="Predicted Next Close",
            )
            ax.axhline(predicted_next_close, color="red", linestyle="--", alpha=0.4)

        if not np.isnan(predicted_next_high):
            ax.scatter(
                [next_date],
                [predicted_next_high],
                color="orange",
                label="Predicted Next High",
            )
            ax.axhline(predicted_next_high, color="orange", linestyle="--", alpha=0.4)

        if not np.isnan(predicted_next_low):
            ax.scatter(
                [next_date],
                [predicted_next_low],
                color="green",
                label="Predicted Next Low",
            )
            ax.axhline(predicted_next_low, color="green", linestyle="--", alpha=0.4)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price (USD)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig


app = shiny.App(app_ui, server)
