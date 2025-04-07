import datetime

import numpy as np
import pandas as pd
import sys 
import csv
import pytz as pytz
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from model.backtester import simple_backtest
from model.helpers import train, predict
from model.preprocessors import process_inputs, process_targets

if __name__ == "__main__":
    # Download price histories from Yahoo Finance
    spy = yf.Ticker(sys.argv[1] if len(sys.argv) > 1 else 'SPY')
    price_series = spy.history(period='max')['Close'].dropna()

    x_df = process_inputs(price_series, window_length=10)

    y_series = process_targets(price_series)

    # Only keep rows in which we have both inputs and data.
    common_index = x_df.index.intersection(y_series.index)
    x_df, y_series = x_df.loc[common_index], y_series.loc[common_index]

    # Train and test model on a walk forward basis with a year gap inbetween
    r2_list = []  # Stores out of sample R Squareds
    corr_list = []  # Stores out of sample correlations
    forecasts = []  # Stores out of sample predictions
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'SPY'
    results_csv = f"results_{ticker.replace('.', '_')}.csv"

    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Year', 'R_Squared', 'Correlation', 'MAE'])
    for training_year in range(2010, datetime.date.today().year + 1):
        training_cutoff = datetime.datetime(training_year, 1, 1, tzinfo=pytz.timezone('America/New_York'))
        test_cutoff = datetime.datetime(training_year + 1, 1, 1, tzinfo=pytz.timezone('America/New_York'))

        # Isolate training data consisting of every data point before `training_year`
        training_x_series = x_df.loc[x_df.index < training_cutoff]
        training_y_series = y_series.loc[y_series.index < training_cutoff]

        # Isolate test data consisting of data points in the year `training_year`
        test_x_series = x_df.loc[(x_df.index >= training_cutoff) & (x_df.index < test_cutoff)]
        test_y_series = y_series.loc[(x_df.index >= training_cutoff) & (x_df.index < test_cutoff)]

        trained_model = train(training_x_series, training_y_series,epochs=1000)

        forecast_series = predict(trained_model, test_x_series)
        results_df = forecast_series.to_frame('Forecast').join(test_y_series.to_frame('Actual')).dropna()
        forecasts.append(results_df)

        # Evaluate forecasts
        results_df.plot.scatter(x='Actual', y='Forecast')
        plt.show()

        r2 = r2_score(results_df['Actual'], results_df['Forecast'])
        r2_list.append(r2)

        corr = results_df.corr().iloc[0, 1]
        corr_list.append(corr)

        mae = mean_absolute_error(results_df['Actual'], results_df['Forecast'])
        print(f"{training_year} R Squared: {r2:.4f}, Correlation: {corr:.4f}, MAE: {mae:.4f}")

# Log to CSV
        with open(results_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([training_year, r2, corr, mae])


    print(f"Average R Squared: {np.average(r2_list):.4f}, Average Correlation: {np.average(corr_list)}")

    # Conduct simple backtest of the strategy
    results_df = pd.concat(forecasts)
    simple_backtest(results_df)
    print(f"\n📁 Saved yearly metrics to {results_csv}")
