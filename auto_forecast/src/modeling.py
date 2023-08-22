import pandas as pd
import numpy as np

# Import sklearn packages
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                            r2_score)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

# import LSTM packages
import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential

from pmdarima.arima import auto_arima

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
            },
            'LSTM': {
                'model': Sequential(),
                'type': 'lstm'
            },
            'ARIMA': {
                'model': 'auto_arima',
                'type': 'arima'
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
        self.train_index = X_train.index

        # train each of the models provided in model_dict 
        for model_name in self.model_list:
            model_type = model_list_options[model_name]['type']
            model_option = model_list_options[model_name]['model']
            if model_type=='regression':
                self.stored_models[model_name]['model'] = self.__fit_regression_model(model_option)
            elif model_type=='lstm':
                self.stored_models[model_name]['model'] = self.__fit_lstm_model(model_option)
            elif model_type=='arima':
                self.stored_models[model_name]['model'] = self.__fit_arima_model(model_name) 
            else: 
                print('Model fit not found.')
    
    def __fit_regression_model(self, model):
        return model.fit(self.X_train, self.y_train)
    
    def __fit_lstm_model(self, model):
        X_train_lstm = self.X_train.values
        X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], 1, X_train_lstm.shape[1])

        model.add(LSTM(4, batch_input_shape=(1, X_train_lstm.shape[1], X_train_lstm.shape[2]), 
                   stateful=True))
        model.add(Dense(1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model.fit(X_train_lstm, self.y_train, epochs=200, batch_size=1, verbose=1, shuffle=False)
    
    def __fit_arima_model(self, model_name):
        model = auto_arima(self.y_train, max_p=12, seasonal=True, m=12)
        self.model_list_options[model_name]['model'] = model
        return model
    
    def predict(self, x_values, y_values=None, scaler=None, print_scores=False):

        model_list_options = self.model_list_options

        self.y_validation_values = y_values
        self.x_predictor_values = x_values
        self.test_index = x_values.index
        self.scaler = scaler

        for model_name in self.model_list:
            if model_list_options[model_name]['type']=='regression':
                predictions = self.__predict_regression_model(model_list_options[model_name]['model'])
            elif model_list_options[model_name]['type']=='lstm':
                predictions = self.__predict_lstm_model(model_list_options[model_name]['model'])
            elif model_list_options[model_name]['type']=='arima':
                predictions = self.__predict_arima_model(model_list_options[model_name]['model'])
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
    
    def __predict_lstm_model(self, model):
        X_test_lstm = self.x_predictor_values.values
        X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], 1, X_test_lstm.shape[1])
        return model.predict(X_test_lstm, batch_size=1)
    
    def __predict_arima_model(self, model):
        num_predictions = len(self.y_validation_values)
        return np.array(model.predict(n_periods=num_predictions))


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
    
    def plot_results(self, model_list=None, xlabel="Date", ylabel="Sales", title="Sales Forecasting Predictions"):
        import parameters as p
        import seaborn as sns

        # concatenate train and test to get all y values
        y_values = self.y_train
        test_index = self.test_index
        train_index = self.train_index
        y_val_values = self.y_validation_values
        stored_models = self.stored_models

        if y_val_values is not None: 
            y_values += y_val_values

        total_index = train_index + test_index

        if model_list == None:
            model_list == list(stored_models.keys())

        fig, ax = plt.subplots(figsize=p.FIG_SIZE)
        sns.lineplot(total_index, y_values, ax=ax, label='Actual', color='blue')
        
        for mod in model_list:
            sns.lineplot(test_index, stored_models[mod]['predictions'], ax=ax, label=f'{mod}_Predictions')

        ax.set(xlabel="Date",
                ylabel="Sales",
                title=title)
        ax.legend()
        sns.despine()
        return fig

        


        

    