import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                            r2_score)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

def isvalid_value(value, values_list):
    if value not in values_list:
        raise ValueError(f"{value} not in the list of available options: {values_list}")
    
class SalesForecasting:

    def __init__(self, model_list):

        self.model_list_options = {
            'LinearRegression': {
                'model': LinearRegression(),
                'type': 'regression'
                },
            'RandomForest': {
                'model': RandomForestRegressor(n_estimators=100, max_depth=20),
                'type': 'regression'
            },
            'XGBoost': {
                'model': XGBRegressor(n_estimators=100, 
                                        learning_rate=0.2, 
                                        objective='reg:squarederror'),
                'type': 'regression'
            }
        }

        if not isinstance(model_list, list):
            raise TypeError('model_dict must be a list.')
        if len(model_list)<1: 
            raise ValueError("model_list must have at least one input.")
        for model in model_list:
            isvalid_value(model, self.model_list_options.keys())
            
        self.model_list = model_list
        self.stored_models = {model_name: {} for model_name in model_list}
    
    def fit(self, X_train, y_train):
        
        model_list_options = self.model_list_options
        self.X_train = X_train
        self.y_train = y_train

        # train each of the models provided in model_dict 
        for model_name in self.model_list:
            if model_list_options[model_name]['type']=='regression':
                self.stored_models[model_name]['model'] = self.__fit_regression_model(model_list_options[model_name]['model'])
            else: 
                print('Model fit not found.')
    
    def __fit_regression_model(self, model):
        return model.fit(self.X_train, self.y_train)
    
    def predict(self, x_values, y_values=None, scaler=None, print_scores=False):

        model_list_options = self.model_list_options

        self.y_validation_values = y_values
        self.x_predictor_values = x_values
        self.scaler = scaler

        for model_name in self.model_list:
            if model_list_options[model_name]['type']=='regression':
                predictions = self.__predict_regression_model(model_list_options[model_name]['model'])
            else: 
                print('Model fit not found.')

            # Undo scaling to compare predictions against original data
            if scaler is not None:
                unscale_values = np.concatenate((x_values.values, predictions.reshape(-1,1)), axis=1)
                predictions = [row[-1] for row in self.__undo_scaling(unscale_values, scaler)]

            self.stored_models[model_name]['predictions'] = predictions
            
            if y_values is not None: 
                unscale_values_true = np.concatenate((x_values.values, y_values), axis=1)
                y_values_unscaled = [row[-1] for row in self.__undo_scaling(unscale_values_true, scaler)]
                self.stored_models[model_name]['true_values'] = y_values_unscaled
                _, _, _ = self.get_scores(predictions, y_values_unscaled, model_name, print_scores)
        
        return self
    
    def __predict_regression_model(self, model):
        return model.predict(self.x_predictor_values)

    def __undo_scaling(self, values, scaler):
        return scaler.inverse(values)
    
    def get_scores(self, y_pred, y_true, model_name=None, print_scores=False):
        rmse = np.sqrt(mean_squared_error(y_pred, y_true))
        mae = mean_absolute_error(y_pred, y_true)
        r2 = r2_score(y_pred, y_true)

        if model_name:
            self.stored_models[model_name]['rmse'] = rmse
            self.stored_models[model_name]['mae'] = mae
            self.stored_models[model_name]['r2'] = r2

        if print_scores:
            if model_name:
                print(f"Scores for {model_name}")
            print("\tRoot mean squared error: {rmse}")
            print("\tMean absolute error: {mae}")
            print("\tR Squared: {r2}")

        return rmse, mae, r2
    

# def lstm_model(train_data, test_data):
#     """Runs a long-short-term-memory nueral net with 2 dense layers. Generates
#     predictions that are then unscaled. Scores are printed and results are
#     plotted and saved.

#     Keyword arguments:
#     -- train_set: dataset used to train the model
#     -- test_set: dataset used to test the model
#     """

#     # Split into X & y and scale data
#     X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)

#     X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
#     X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#     # Build LSTM
#     model = Sequential()
#     model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
#     model.add(Dense(1))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, shuffle=False)
#     predictions = model.predict(X_test, batch_size=1)

#     # Undo scaling to compare predictions against original data
#     original_df = load_data('../data/monthly_data.csv')
#     unscaled = undo_scaling(predictions, X_test, scaler_object, lstm=True)
#     unscaled_df = predict_df(unscaled, original_df)

#     # print scores and plot results
#     get_scores(unscaled_df, original_df, 'LSTM')
#     plot_results(unscaled_df, original_df, 'LSTM')

# def sarimax_model(data):
#     """Runs an arima model with 12 lags and yearly seasonal impact. Generates
#     dynamic predictions for last 12 months. Prints and saves scores and plots
#     results.
#     """
#     # Model
#     sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(12, 0, 0),
#                                     seasonal_order=(0, 1, 0, 12),
#                                     trend='c').fit()

#     # Generate predictions
#     start, end, dynamic = 40, 100, 7
#     data['pred_value'] = sar.predict(start=start, end=end, dynamic=dynamic)

#     # Generate predictions dataframe
#     original_df = load_data('../data/monthly_data.csv')
#     unscaled_df = predict_df(data, original_df)

#     # print scores and plot results
#     get_scores(unscaled_df, original_df, 'ARIMA')
#     plot_results(unscaled_df, original_df, 'ARIMA')

# def sarimax_model(data):
#     """Runs an arima model with 12 lags and yearly seasonal impact. Generates
#     dynamic predictions for last 12 months. Prints and saves scores and plots
#     results.
#     """
#     # Model
#     sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(12, 0, 0),
#                                     seasonal_order=(0, 1, 0, 12),
#                                     trend='c').fit()

#     # Generate predictions
#     start, end, dynamic = 40, 100, 7
#     data['pred_value'] = sar.predict(start=start, end=end, dynamic=dynamic)

#     # Generate predictions dataframe
#     original_df = load_data('../data/monthly_data.csv')
#     unscaled_df = predict_df(data, original_df)

#     # print scores and plot results
#     get_scores(unscaled_df, original_df, 'ARIMA')
#     plot_results(unscaled_df, original_df, 'ARIMA')