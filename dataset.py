import datetime
from enum import Enum
import math
import os
import pandas as pd
import random
import torch

class DatasetType(Enum):
    train = 0
    validation = 1
    test = 2
    

class PrecipitationClassificationDataset(torch.utils.data.Dataset):
    # Datasets utilized for this project contain years of data with data for the summer months removed
    def __init__(self, folder: str, file_label: str, lookback: int, dataset_type: DatasetType, num_forecast_steps=1):
        """file_label identifies each draw of data points within each combination, e.g. '10-sites__50-miles__draw3'"""
        super().__init__()
        self.lookback = lookback
        self.num_forecast_steps = num_forecast_steps
        if dataset_type == DatasetType.train:
            filename = file_label + "__train.csv"
        elif dataset_type == DatasetType.validation:
            filename = file_label + "__valid.csv"
        else:
            filename = file_label + "__test.csv"
        filepath = os.path.join(os.getcwd(), folder, filename)

        # Load the dataframe and converting it into a usable format for training
        df = pd.read_csv(filepath, index_col=0) # Use the first column as the index
        # Convert string indicies to timestamps
        date_index = list(map(pd.to_datetime, df.index.to_list()))
        df.index = date_index
        # Transform variables (first training_proportion*100 percent of the data used as training data)
        years = df.index.year.unique()
        invalid_timesteps = []
        for year in years:
                # Since the data for this project excludes the summer months, the time series is broken
                # 31 days in may, num_forecast_steps - 1 is used since the forecast includes the precipitation for the current time step
                # num_forecast_steps > 0, lookback > 0
                may_invalid_timestep_start = datetime.datetime(year, 5, 31 - ((num_forecast_steps - 1) // 24), 23 - ((num_forecast_steps - 1) % 24))
                may_invalid_timestep_end = datetime.datetime(year, 6, 1)
                may_invalid_timesteps = pd.date_range(start=may_invalid_timestep_start, end=may_invalid_timestep_end, freq="h", inclusive="neither")

                sept_invalid_timestep_start = datetime.datetime(year, 9, 1)
                # Again, - 1 is present since the lookback includes the current time step
                sept_invalid_timestep_end = datetime.datetime(year, 9, 1 + ((lookback - 1) // 24), (lookback - 1) % 24)
                sept_invalid_timesteps = pd.date_range(start=sept_invalid_timestep_start, end=sept_invalid_timestep_end, freq="h", inclusive="left")
                
                invalid_timesteps += (list(may_invalid_timesteps) + list(sept_invalid_timesteps))

        i = 0 # Index used within df
        dataset = []
        while i + lookback + num_forecast_steps - 1 <= len(df): # The condition on the while cuts out the last few samples, as there aren't enough forecast steps
            df_slice = df.iloc[i : i+lookback+num_forecast_steps-1] # slice containing rows i to i+lookback+num_forecast_steps-2
            timestep = df.index[i + lookback - 1] # Index of the time step to forecast for

            # Check for bad sequences
            if timestep not in invalid_timesteps:
                feature_slice = df_slice.iloc[:lookback, :-1] # Exclude last column, which is the target
                target_slice = df_slice.iloc[lookback-1:, -1] # Only include last (target) column
                feature_tensor = torch.tensor(feature_slice.values.tolist())
                target_tensor = torch.tensor(target_slice.values.tolist())
                dataset.append((timestep, feature_tensor, target_tensor))
            i += 1
                
        self.ds = dataset # Array with elements of the form (timestep, features, targets); both features and targets are tensors


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index][1], self.ds[index][2]
