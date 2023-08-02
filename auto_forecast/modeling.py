import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                            r2_score)

from auto_forecast.data_processing import (scale_data)
from auto_forecast.plotting import (plot_results)


def predict_df(unscaled_predictions, original_df):
    """Generates a dataframe that shows the predicted sales for each month
    for plotting results.

    Keyword arguments:
    -- unscaled_predictions: the model predictions that do not have min-max or
                             other scaling applied
    -- original_df: the original monthly sales dataframe
    """
    #create dataframe that shows the predicted sales
    result_list = []
    sales_dates = list(original_df[-13:].date)
    act_sales = list(original_df[-13:].sales)

    for index in range(len(unscaled_predictions)):
        result_dict = {
            'pred_value': int(
                unscaled_predictions[index][0] + act_sales[index]
            ),
            'date': sales_dates[index + 1],
        }
        result_list.append(result_dict)

    return pd.DataFrame(result_list)

def get_scores(unscaled_df, original_df, model_name):
    """Prints the root mean squared error, mean absolute error, and r2 scores
    for each model. Saves all results in a model_scores dictionary for
    comparison.

    Keyword arguments:
    -- unscaled_predictions: the model predictions that do not have min-max or
                             other scaling applied
    -- original_df: the original monthly sales dataframe
    -- model_name: the name that will be used to store model scores
    """
    rmse = np.sqrt(mean_squared_error(original_df.sales[-12:], unscaled_df.pred_value[-12:]))
    mae = mean_absolute_error(original_df.sales[-12:], unscaled_df.pred_value[-12:])
    r2 = r2_score(original_df.sales[-12:], unscaled_df.pred_value[-12:])
    model_scores[model_name] = [rmse, mae, r2]

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")



def regressive_model(train_data, test_data, model, model_name):
    """Runs regressive models in SKlearn framework. First calls scale_data
    to split into X and y and scale the data. Then fits and predicts. Finally,
    predictions are unscaled, scores are printed, and results are plotted and
    saved.

    Keyword arguments:
    -- train_set: dataset used to train the model
    -- test_set: dataset used to test the model
    -- model: the sklearn model and model arguments in the form of
              model(kwarga)
    -- model_name: the name that will be used to store model scores and plotting
    """

    # Split into X & y and scale data
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data,
                                                                 test_data)
    # Run sklearn models
    mod = model
    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)

    # Undo scaling to compare predictions against original data
    original_df = load_data('../data/monthly_data.csv')
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)

    # print scores and plot results
    get_scores(unscaled_df, original_df, model_name)
    plot_results(unscaled_df, original_df, model_name)

def lstm_model(train_data, test_data):
    """Runs a long-short-term-memory nueral net with 2 dense layers. Generates
    predictions that are then unscaled. Scores are printed and results are
    plotted and saved.

    Keyword arguments:
    -- train_set: dataset used to train the model
    -- test_set: dataset used to test the model
    """

    # Split into X & y and scale data
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Build LSTM
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, shuffle=False)
    predictions = model.predict(X_test, batch_size=1)

    # Undo scaling to compare predictions against original data
    original_df = load_data('../data/monthly_data.csv')
    unscaled = undo_scaling(predictions, X_test, scaler_object, lstm=True)
    unscaled_df = predict_df(unscaled, original_df)

    # print scores and plot results
    get_scores(unscaled_df, original_df, 'LSTM')
    plot_results(unscaled_df, original_df, 'LSTM')

def sarimax_model(data):
    """Runs an arima model with 12 lags and yearly seasonal impact. Generates
    dynamic predictions for last 12 months. Prints and saves scores and plots
    results.
    """
    # Model
    sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(12, 0, 0),
                                    seasonal_order=(0, 1, 0, 12),
                                    trend='c').fit()

    # Generate predictions
    start, end, dynamic = 40, 100, 7
    data['pred_value'] = sar.predict(start=start, end=end, dynamic=dynamic)

    # Generate predictions dataframe
    original_df = load_data('../data/monthly_data.csv')
    unscaled_df = predict_df(data, original_df)

    # print scores and plot results
    get_scores(unscaled_df, original_df, 'ARIMA')
    plot_results(unscaled_df, original_df, 'ARIMA')

def main():
    """Calls all functions to load data, run regression models, run lstm model,
    and run arima model.
    """
    # Regression models
    model_df = load_data('../data/model_df.csv')
    train, test = tts(model_df)

    # Sklearn
    regressive_model(train, test, LinearRegression(), 'LinearRegression')
    regressive_model(train, test, RandomForestRegressor(n_estimators=100,
                                                        max_depth=20),
                     'RandomForest')
    regressive_model(train, test, XGBRegressor(n_estimators=100,
                                               learning_rate=0.2,
                                               objective='reg:squarederror'),
                     'XGBoost')
    # Keras
    lstm_model(train, test)

    # Arima
    ts_data = load_data('../data/arima_df.csv').set_index('date')
    ts_data.index = pd.to_datetime(ts_data.index)

    sarimax_model(ts_data)