from sklearn.preprocessing import MinMaxScaler

def difference_data(data, value_col):
    if value_col not in data.columns:
        raise ValueError("value_col must exist within the data columns.")

    new_col_name = f'{value_col}_differenced'
    data[new_col_name] = data[value_col].diff()
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
    return data.iloc[:-test_size], data.iloc[:-test_size:]

def reshape_data(data):
    return data.reshape(data.shape[0], data.shape[1])

def min_max_built_in_scaler(train_set, test_set):
    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    
    # reshape training set
    train_set = reshape_data(train_set)
    train_set_scaled = scaler.transform(train_set)
    
    # reshape test set
    test_set = reshape_data(test_set)
    test_set_scaled = scaler.transform(test_set)
    
    return train_set_scaled, test_set_scaled, scaler

def get_x_y(data, target_col):
    return data.drop(target_col), data[[target_col]]