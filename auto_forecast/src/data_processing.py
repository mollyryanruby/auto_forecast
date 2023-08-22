# standard data manipulation imports
import pandas as pd

# import scaling functions
from sklearn.preprocessing import MinMaxScaler


class DataScaler:
    """
    Scale the data using a MinMaxScaler.
    """
    def __ini__(self):
        self.scaler = None

    def __separate_columns(self, data):
        """
        Separate the data into date columns and non-date columns.

        Args:
            data (pd.DataFrame): data to separate
        
        Returns:
            pd.DataFrame, np.array: date columns, non-date columns
        """
        date_cols = [col for col in data.columns if data.dtypes[col]=='<M8[ns]']
        data_to_store = data[date_cols]
        data_to_scale = data[[col for col in data.columns if col not in date_cols]]
        data_to_scale_values = data_to_scale.values
        return data_to_store, data_to_scale_values

    def fit(self, data):
        """
        Fit the scaler to the data.

        Args:
            data (pd.DataFrame): data to fit the scaler to
        
        Returns: 
            MinMaxScaler: scaler object
        """
        # separate anything that is a date columns
        
        _, data_to_scale_values = self.__separate_columns(data)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = scaler.fit(data_to_scale_values)
        return self.scaler

    def transform(self, data):
        """
        Transform the data using the scaler.

        Args:
            data (pd.DataFrame): data to transform

        Returns:
            pd.DataFrame: transformed data
        """
        if not isinstance(self.scaler, MinMaxScaler):
            raise ValueError("Scaler object must be fit before transformed.")
        
        data_to_store, data_to_scale_values = self.__separate_columns(data)
        scaled_data = pd.DataFrame(self.scaler.transform(data_to_scale_values))
        reconciled_data = pd.concat((data_to_store.reset_index(drop=True), scaled_data.reset_index(drop=True)), axis=1)
        reconciled_data.columns = data.columns
        return reconciled_data
    
    def fit_transform(self, data):
        """
        Fit and transform the data.

        Args:
            data (pd.DataFrame): data to fit and transform

        Returns:
            pd.DataFrame: transformed data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse(self, values):
        """
        Inverse the values using the scaler.

        Args:
            values (np.array): values to inverse

        Returns:
            np.array: inverse values
        """
        return self.scaler.inverse_transform(values)
    

def aggregate_by_time(data, date_col, resample_freq, aggregate):
    """
    Aggregate the data by the resample_freq using the aggregate function.

    Args:
        data (pd.DataFrame): data to aggregate
        date_col (str): column name of the date column
        resample_freq (str): frequency to resample the data
        aggregate (str): aggregate function to use

    Returns:
        pd.DataFrame: aggregated data
    """
    data[date_col] = pd.to_datetime(data[date_col])
    resampled = data.set_index(date_col).resample(resample_freq).agg(aggregate)
    return resampled.reset_index()

def difference_data(data, date_col, value_col, diff_value_col_name=None):
    """
    Difference the data by the value_col.

    Args:
        data (pd.DataFrame): data to difference
        date_col (str): column name of the date column
        value_col (str): column name of the value column
        diff_value_col_name (str, optional): column name of the differenced value column. Defaults to None.

    Returns:
        pd.DataFrame: differenced data
    """
    if value_col not in data.columns:
        raise ValueError("value_col must exist within the data columns.")
    if not diff_value_col_name:
        diff_value_col_name = f'{value_col}_differenced'

    data = data.sort_values(by=date_col).reset_index(drop=True)

    data[diff_value_col_name] = data[value_col].diff()
    return data
    
def create_lag_data(data, date_col, value_col, lags):
    """
    Create lag columns for the data.

    Args:
        data (pd.DataFrame): data to create lags for
        date_col (str): column name of the date column
        value_col (str): column name of the value column
        lags (int): number of lags to create
    
    Returns:
        pd.DataFrame: data with lag columns 
    """
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
    """
    Create train and test data.

    Args:
        data (pd.DataFrame): data to split
        test_size (int): size of the test data
    
    Returns:
        pd.DataFrame: train data
    """
    return data.iloc[:-test_size], data.iloc[-test_size:]

def get_x_y(self, data):
    """
    Get the x and y data from the data.

    Args:
        data (pd.DataFrame): data to get x and y from

    Returns:
        pd.DataFrame, pd.DataFrame: x and y data
    """    
    return data[self.x_cols], data[[self.target_col]]


