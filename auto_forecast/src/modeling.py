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

import parameters as p
import matplotlib.pyplot as plt
import seaborn as sns

def isvalid_value(value, values_list):

    if value not in values_list:
        raise ValueError(f"{value} not in the list of available options: {values_list}")
    
class SalesForecasting:
    """
    SalesForecasting class to train and predict sales using a variety of models. 
    """

    def __init__(self, model_list):
        """
        Initialize the SalesForecasting class with a list of models to train and predict.

        Args:
            model_list (list): list of models to train and predict. Options include:
                - LinearRegression
                - RandomForest
                - XGBoost
                - LSTM
                - ARIMA

        Returns:
            None
        """

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
        """
        Fit the models in model_dict to the training data.

        Args:
            X_train (pd.DataFrame): training data exogonous features for the model
            y_train (pd.Series): training data target for the model
        
        Returns:
            None
        """
        
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
        """
        Fit a regression model to the training data.

        Args:
            model (sklearn model): sklearn model to fit to the training data
        
        Returns:
            model (sklearn model): fitted sklearn model
        """
        return model.fit(self.X_train, self.y_train)
    
    def __fit_lstm_model(self, model):
        """
        Fit an LSTM model to the training data.

        Args:
            model (keras model): keras model to fit to the training data
        
        Returns:
            model (keras model): fitted keras model
        """

        X_train_lstm = self.X_train.values
        X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], 1, X_train_lstm.shape[1])

        model.add(LSTM(4, batch_input_shape=(1, X_train_lstm.shape[1], X_train_lstm.shape[2]), 
                   stateful=True))
        model.add(Dense(1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model.fit(X_train_lstm, self.y_train, epochs=200, batch_size=1, verbose=1, shuffle=False)
    
    def __fit_arima_model(self, model_name):
        """
        Fit an ARIMA model to the training data.

        Args:
            model_name (str): name of the model to fit to the training data
        
        Returns:
            model (pmdarima model): fitted pmdarima model
        """
        model = auto_arima(self.y_train, max_p=12, seasonal=True, m=12)
        self.model_list_options[model_name]['model'] = model
        return model
    
    def predict(self, x_values, y_values=None, scaler=None, print_scores=False):
        """
        Predict values using the models in model_dict.

        Args:
            x_values (pd.DataFrame): exogenous features to predict on
            y_values (pd.Series): target values to compare predictions against
            scaler (sklearn scaler): scaler used to scale the data
            print_scores (bool): whether to print the scores for each model
        
        Returns:
            self (SalesForecasting): self with updated predictions
        """

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

                # unscale the training set as well for reference
                training_set = pd.concat((self.X_train, self.y_train), axis=1).values
                unscaled_train = self.__undo_scaling(training_set, scaler)
                unscaled_y_train = [row[-1] for row in unscaled_train]
                unscaled_x_train = [row[:-1] for row in unscaled_train]
                self.unscaled_y_train = unscaled_y_train
                self.unscaled_x_train = unscaled_x_train

                if y_values is not None: 
                    unscale_values_true = np.concatenate((x_values.values, y_values), axis=1)
                    y_values_unscaled = [row[-1] for row in self.__undo_scaling(unscale_values_true, scaler)]
                    self.y_values_unscaled = y_values_unscaled
                    self.stored_models[model_name]['true_values'] = y_values_unscaled
                    _, _, _ = self.get_scores(predictions, y_values_unscaled, model_name, print_scores)
            
            elif y_values is not None: 
                self.stored_models[model_name]['true_values'] = y_values.iloc[:, 0].to_list()

            self.stored_models[model_name]['predictions'] = predictions                    

        
        return self
    
    def __predict_regression_model(self, model):
        """
        Predict values using a regression model.
        
        Args:
            model (sklearn model): sklearn model to predict with

        Returns:
            predictions (np.array): array of predictions
        """
        return model.predict(self.x_predictor_values)
    
    def __predict_lstm_model(self, model):
        """
        Predict values using an LSTM model.

        Args:
            model (keras model): keras model to predict with

        Returns:
            predictions (np.array): array of predictions
        """
        X_test_lstm = self.x_predictor_values.values
        X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], 1, X_test_lstm.shape[1])
        return model.predict(X_test_lstm, batch_size=1)
    
    def __predict_arima_model(self, model):
        """
        Predict values using an ARIMA model.

        Args:
            model (pmdarima model): pmdarima model to predict with
        Returns: 
            predictions (np.array): array of predictions
        """
        num_predictions = len(self.y_validation_values)
        return np.array(model.predict(n_periods=num_predictions))


    def __undo_scaling(self, values, scaler):
        """
        Undo scaling on a set of values.

        Args:
            values (np.array): array of values to unscale
            scaler (sklearn scaler): scaler to use to unscale the values

        Returns:
            unscaled_values (np.array): array of unscaled values
        """
        return scaler.inverse(values)
    
    def get_scores(self, y_pred, y_true, model_name=None, print_scores=False):
        """
        Get the scores for a model. Scores include RMSE, MAE, and R2.

        Args:
            y_pred (np.array): array of predicted values
            y_true (np.array): array of true values
            model_name (str): name of the model to get scores for
            print_scores (bool): whether to print the scores for the model
        
        Returns:
            rmse (float): root mean squared error
            mae (float): mean absolute error
            r2 (float): r squared
        """
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
    
    def plot_results(self, model_list=None, figsize=p.FIG_SIZE, xlabel="Date", ylabel="Sales", title="Sales Forecasting Predictions"):
        """
        Plot the results of the predictions against the actual values.
        Generates a timeseries for predictions from each model in model_dict.

        Args:
            model_list (list): list of models to plot. If None, plots all models in model_dict
            figsize (tuple): tuple of figure size
            xlabel (str): label for x axis
            ylabel (str): label for y axis
            title (str): title for the plot
        
        Returns:
            fig (matplotlib figure): figure with the plot
        """

        # concatenate train and test to get all y values
        test_index = list(self.test_index)
        train_index = list(self.train_index)
        y_val_values = self.y_validation_values
        stored_models = self.stored_models
        y_values = self.y_train
        if (y_val_values is not None) & (self.scaler is not None): 
            y_values = np.concatenate((self.unscaled_y_train, self.y_values_unscaled))
        elif (y_val_values is not None):
            y_values = np.concatenate((y_values, y_val_values))
        elif (self.scaler is not None):
            y_values = self.unscaled_y_train

        total_index = train_index + test_index

        if model_list is None:
            model_list = list(stored_models.keys())

        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(x=total_index, y=y_values, ax=ax, label='Actual', color=p.COLORS[0])
        
        # need to use total index to plot on same visual
        # to do so, must have y be teh same length for both actuals and predictions
        # so we fill y in predictions train period with null
        filler_y_values = [np.nan] * len(train_index)
        for i, mod in enumerate(model_list):
            color = p.COLORS[i+1]
            sns.lineplot(
                x=total_index, 
                y=filler_y_values + stored_models[mod]['predictions'], 
                ax=ax, 
                label=f'{mod}_Predictions',
                color=color
                )

        ax.set(xlabel=xlabel,
                ylabel=ylabel,
                title=title)
        ax.legend()
        sns.despine()
        return fig
    
    def plot_errs(self, figsize=(13,3)):
        """
        Plot the errors for each model in model_dict. Errors include RMSE, MAE, and R2.

        Args:
            figsize (tuple): tuple of figure size
        
        Returns:
            fig (matplotlib figure): figure with the plot
        """
        output_df = pd.DataFrame(self.stored_models).T
        errs = ['rmse', 'mae', 'r2']
        
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        for i, err in enumerate(errs):
            err_plot = sns.barplot(y=output_df[err], x=output_df.index, color=p.COLORS[i], ax=ax[i])
            err_plot.tick_params(labelrotation=45)
            err_plot.set_title(err)
            err_plot.set_ylabel('') 

        return fig

        


        

    