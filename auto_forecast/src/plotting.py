# plotting code
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import graphics

# standard data manipulation imports
import pandas as pd
import numpy as np

# intra-package code
import parameters as p

def plot_periodic_values_hist(data, value_col, figsize=p.FIG_SIZE, color=p.PLOT_COLOR_1):
    if value_col not in data.columns:
        raise ValueError("value_col must exist within the data columns.")

    fig, ax = plt.subplots(figsize=figsize)
    plt.hist(data[value_col], color=color)
    
    ax.set(
        xlabel = f"{value_col} per day",
        ylabel = "count",
        title = f"distrobution of {value_col} per day"
        )
    return fig, ax
    

def plot_values_per_group(data, value_col, group_cols, figsize=p.FIG_SIZE, fill_color=p.PLOT_COLOR_1):
    if value_col not in data.columns:
        raise ValueError("value_col must exist within the data columns.")
    if not isinstance(group_cols, str) and not isinstance(group_cols, list):
        raise KeyError("group_cols must be a stirng or list.")
    if isinstance(group_cols, str):
        group_cals = list(group_cols)
    
    data[group_cols] = data[group_cols].astype(str)
    data['group_id'] = data[group_cols].agg('-'.join, axis=1) if len(group_cols)>1 else data[group_cols]

    by_group = data.groupby('group_id')[value_col].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=by_group['group_id'], y=by_group[value_col], color=fill_color)
    
    ax.set(
        xlabel = "group id",
        ylabel = "summed value for the group",
        title = f"{value_col} per group"
        )
    
    sns.despine()
    return fig, ax

def plot_time_series(data, date_col, value_col, mean_freq=None, figsize=p.FIG_SIZE, color=p.PLOT_COLOR_1):
    if date_col not in data.columns:
        raise ValueError("date_col must exist within the data columns.")
    if value_col not in data.columns:
        raise ValueError("value_col must exist within the data columns.")

    data[date_col] = pd.to_datetime(data[date_col])
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        x=data[date_col], 
        y=data[value_col], 
        ax=ax, 
        color=color, 
        label='Total')

    ax.set(xlabel = "date",
            ylabel = "values",
            title = "total and mean timeplot")
    
    sns.despine()
    
    # optionally plot mean trend 
    if mean_freq is not None: 
        average_data = data.groupby(pd.Grouper(key=date_col, freq=mean_freq))[value_col].mean().reset_index()

        sns.lineplot(
            x=average_data[date_col], 
            y=average_data[value_col],
            ax=ax, 
            color='red', 
            label='Average'
            )   
        
    

def plt_acf_pcf(data, date_col, value_col, lags=None, color='mediumblue', figsize=p.FIG_SIZE, **kwargs):
    if value_col not in data.columns:
        raise ValueError("value_col must exist within the data columns.")
    
    # Convert dataframe to datetime index
    dt_data = data.set_index(date_col)[[value_col]].dropna()
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    
    dt_data.plot(ax=ax[0], color=color)
    graphics.plot_acf(dt_data[value_col], lags=lags, ax=ax[1], color=color)
    graphics.plot_pacf(dt_data[value_col], lags=lags, ax=ax[2], color=color)

    sns.despine()
    plt.tight_layout()

def plot_lag_cols(data, date_col, value_col, lag_col_root, num_lags=None, figsize=p.FIG_SIZE):
    
    all_lag_cols = [col for col in data.columns if lag_col_root in col]
    if not num_lags:
        num_lags=len(all_lag_cols)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data[date_col], data[value_col], label='original')
    
    for lag_col in all_lag_cols[:num_lags]:
        ax.plot(data[date_col], data[lag_col], label=lag_col)
    
    plt.legend()

def visualize_train_test(train, test, date_col, value_col, figsize=p.FIG_SIZE):
    fig, ax = plt.subplots(figsize=(6, 3)) 

    full_data = pd.concat((train, test))
    sns.lineplot(x=train[date_col], y=train[value_col], ax=ax, label='train')
    sns.lineplot(x=test[date_col], y=test[value_col], ax=ax, label='test')
    sns.despine()

def plot_results(results, original_df, model_name, figsize=p.FIG_SIZE, color1=p.PLOT_COLOR_1, color2=p.PLOT_COLOR_2):
    """Plots predictions over original data to visualize results. Saves each
    plot as a png.

    Keyword arguments:
    -- results: a dataframe with unscaled predictions
    -- original_df: the original monthly sales dataframe
    -- model_name: the name that will be used in the plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(original_df.date, original_df.sales, data=original_df, ax=ax,
                 label='Original', color=color1)
    sns.lineplot(results.date, results.pred_value, data=results, ax=ax,
                 label='Predicted', color=color2)
    ax.set(xlabel="Date",
           ylabel="Sales",
           title=f"{model_name} Sales Forecasting Prediction")
    ax.legend()
    sns.despine()
