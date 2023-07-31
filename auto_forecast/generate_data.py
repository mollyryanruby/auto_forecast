import pandas as pd
import numpy as np

from datetime import timedelta
from random import randint


class DataSamples:
    def __init__(self, start_date, num_periods, granularity, empty=False):
        self.start_date=start_date
        self.num_periods=num_periods
        self.granularity=granularity
        self.empty=empty

        if not isinstance(self.start_date, str):
            raise ValueError("start_date must be of type string.")
        if not isinstance(self.num_periods, int):
            raise ValueError("num_periods must be of type int.")

        acceptable_granularity_values = [
            'days', 
            'seconds', 
            'microseconds', 
            'milliseconds', 
            'minutes', 
            'hours', 
            'weeks'
            ]

        if granularity not in acceptable_granularity_values:
            raise ValueError(f"granularity must be a value in {acceptable_granularity_values}")

    def __get_list_of_dates(self):
        '''
        Create list of dates
        '''
        self.start_date = pd.to_datetime(self.start_date)

        return [
            self.start_date + timedelta(**{self.granularity: i}) 
            for i 
            in range(self.num_periods)
        ]
    

    def __get_data_values(self):
        """returns random data.
        """
        return [randint(self.min_given, self.max_given) for _ in self.num_periods]

    def __generate_stats(self):
        self.all_dates = pd.to_datetime(self.data[self.date_col])
        date_diff = self.all_dates.max() - self.all_dates.min()
        self.number_of_days = date_diff.days
        self.number_of_years = self.number_of_days.days / 365
        self.max_value = self.data[self.value_col].max()
        self.min_value = self.data[self.value_col].min()
        self.mean_value = self.data[self.value_col].mean()

    def __get_stats(self):
        print("Date metrics")
        print(f"\t>Total number of days: {self.number_of_days}")
        print(f"\t>Total number of years: {self.number_of_years}")
        print(f"\nValue Metrics")
        print(f"\t>Min value: {self.min_value}")
        print(f"\t>Max value: {self.max_value}")
        print(f"\t>Average value: {self.mean_value}")

    def create_sample_data(
            self, 
            min = 0,
            max = 100,
            date_col='date',
            value_col='value'
            ):
        
        self.min_given = min
        self.max_given = max
        self.date_col = date_col
        self.value_col = value_col
        
        if not self.empty:
            values = self.get_data_values(min, max)

        else: 
            values = [np.nan] * self.num_periods

        self.data = pd.DataFrame({
            self.date_col: self.get_list_of_dates(),
            self.value_col: values
            })
        
        return self.data