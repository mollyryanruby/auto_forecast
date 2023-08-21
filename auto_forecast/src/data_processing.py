# standard data manipulation imports
import pandas as pd

# import scaling functions
from sklearn.preprocessing import MinMaxScaler

def aggregate_by_time(data, date_col, resample_freq, aggregate):
    data[date_col] = pd.to_datetime(data[date_col])
    resampled = data.set_index(date_col).resample(resample_freq).agg(aggregate)
    return resampled.reset_index()

def difference_data(data, date_col, value_col, diff_value_col_name=None):
    if value_col not in data.columns:
        raise ValueError("value_col must exist within the data columns.")
    if not diff_value_col_name:
        diff_value_col_name = f'{value_col}_differenced'

    data = data.sort_values(by=date_col).reset_index(drop=True)

    data[diff_value_col_name] = data[value_col].diff()
    return data
    
def create_lag_data(data, date_col, value_col, lags):
    #create dataframe for transformation from time series to supervised
    if not isinstance(lags, int):
        raise TypeError("lags must be an integer.")
    if date_col not in data.columns:
        raise ValueError("date_col must exist within the data columns.")
    if value_col not in data.columns:
        raise ValueError("value_col must exist within the data columns.")

    data = data.sort_values(by=date_col)

    #create column for each lag
    for i in range(1, lags):
        data[f'lag_{str(i)}'] = data[value_col].shift(i)

    #drop null values from creating lags 
    return data.dropna()

def create_train_test(data, test_size):
    return data.iloc[:-test_size], data.iloc[-test_size:]

def get_x_y(self, data):
        return data[self.x_cols], data[[self.target_col]]


class DataScaler:
    def __ini__(self):
        self.scaler = None

    def __separate_columns(self, data):
        date_cols = [col for col in data.columns if data.dtypes[col]=='<M8[ns]']
        data_to_store = data[date_cols]
        data_to_scale = data[[col for col in data.columns if col not in date_cols]]
        data_to_scale_values = data_to_scale.values
        return data_to_store, data_to_scale_values

    def fit(self, data):
        # separate anything that is a date columns
        
        _, data_to_scale_values = self.__separate_columns(data)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = scaler.fit(data_to_scale_values)
        return self.scaler

    def transform(self, data):
        if not isinstance(self.scaler, MinMaxScaler):
            raise ValueError("Scaler object must be fit before transformed.")
        
        data_to_store, data_to_scale_values = self.__separate_columns(data)
        scaled_data = pd.DataFrame(self.scaler.transform(data_to_scale_values))
        reconciled_data = pd.concat((data_to_store.reset_index(drop=True), scaled_data.reset_index(drop=True)), axis=1)
        reconciled_data.columns = data.columns
        return reconciled_data
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse(self, values):
        return self.scaler.inverse_transform(values)