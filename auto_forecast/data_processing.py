

    
def difference_data(df, value_col):
    new_col_name = value_col + '_differenced'
    df[new_col_name] = df[value_col].diff()
    return df
    
        