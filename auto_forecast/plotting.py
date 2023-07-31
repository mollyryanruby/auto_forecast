# plotting code
import matplotlib.pyplot as plt
import seaborn as sns

# standard data manipulation imports
import pandas as pd

# intra-package code
import parameters as p

def periodic_values_hist(df, value_col='value', figsize=p.FIG_SIZE):

    fig, ax = plt.subplots(figsize=figsize)
    plt.hist(df[value_col], color='mediumblue')
    
    ax.set(
        xlabel = f"{value_col} per day",
        ylabel = "count",
        title = f"distrobution of {value_col} per day"
        )
    return fig, ax
    

def values_per_group(df, value_col, group_cols, figsize=p.FIG_SIZE, fill_color='mediumblue'):
    
    if len(group_cols)<1:
        raise ValueError("group_cols requires at least one column.")
    

    df['group_id'] = df[group_cols].agg('-'.join, axis=1)
    by_group = df.groupby('group_id')[value_col].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(df['group_id'], df[value_col], color=fill_color)
    
    ax.set(
        xlabel = "group id",
        ylabel = "summed value for the group",
        title = f"{value_col} per group"
        )
    
    sns.despine()
    return fig, ax

def standard_time_plot(df, date_col, value_col, mean_freq, figsize=p.FIG_SIZE):


    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        df[date_col], 
        df[value_col], 
        data=df, 
        ax=ax, 
        color='mediumblue', 
        label='Total')
    
    df[date_col] = pd.to_datetime(df[date_col])
    average_df = df.groupby(pd.Grouper(key=date_col, freq=mean_freq)).mean().reset_index()

    sns.lineplot(
        average_df[date_col], 
        average_df[value_col], 
        data=average_df, 
        ax=ax, 
        color='red', 
        label='Average'
        )   
    
    ax.set(xlabel = "date",
           ylabel = "values",
           title = "total and mean timeplot")
    
    sns.despine()
