import datetime
import os
import pandas as pd
import torch

class WeatherDataset(torch.utils.data.Dataset):
    # Datasets utilized for this project contain 2 years of data with data for the summer months removed.
    def __init__(self, folder: str, file: str, lookback: int, num_forecast_steps: int):
        super().__init__()
        self.lookback = lookback
        self.num_forecast_steps = num_forecast_steps
        path = os.getcwd() + "/" + folder + "/" + file
        # Loading the dataframe and converting it into a usable format for training
        df = pd.read_csv(path, index_col=0) # Use the first column as the index
        # Convert string indicies to timestamps
        date_index = list(map(pd.to_datetime, df.index.to_list()))
        df.index = date_index
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
        n = 0 # Index of processed df;
        dataset = {}
        while i + lookback + num_forecast_steps - 1 <= len(df):
            df_slice = df.iloc[i : i+lookback+num_forecast_steps-1] # slice containing rows i to i+lookback+num_forecast_steps-2
            timestep = df.index[i + lookback - 1] # Index of the time step to forecast for

            # Check for bad sequences
            if timestep  not in invalid_timesteps:
                feature_slice = df_slice.iloc[:lookback, :-1] # Exclude last column, which is the target
                target_slice = df_slice.iloc[lookback-1:, -1] # Only include last (target) column
                feature_tensor = torch.tensor(feature_slice.values.tolist())
                # feature_tensor = feature_tensor.reshape(-1,)
                target_tensor = torch.tensor(target_slice.values.tolist())
                dataset[n] = (timestep, feature_tensor, target_tensor)
                n += 1
            
            i += 1
            
                

        self.ds = dataset # Dictionary of the form {index: (timestep, features, targets)}


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index][1], self.ds[index][2]


if __name__ == "main":
    wds = WeatherDataset("CSV_Datasets", "16-sites__100-miles.csv", 24, 6)