import datetime
import os
import pandas as pd
import torch

class WeatherDataset(torch.utils.data.Dataset):
    # Datasets utilized for this project contain 2 years of data with data for the summer months removed.
    def __init__(self, folder, file, lookback, num_forecast_steps):
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
        i = 0 # Index used within df
        n = 0 # Index of processed df;
        dataset = {}
        while i + lookback + num_forecast_steps - 1 <= len(df):
            df_slice = df.iloc[i : i+lookback+num_forecast_steps-1] # slice containing rows i to i+lookback+num_forecast_steps-2
            bad_sequence = False

            # Check for bad sequences
            for year in years:
                # Since the data for this project excludes the summer months, the time series is broken
                may_border = datetime.datetime(year, 5, 31, 23)
                sept_border = datetime.datetime(year, 9, 1, 0)
                if (may_border in df_slice.index) and (sept_border in df_slice.index):
                    # Resulting sequence will contain data from both May and September, skip such sequences
                    bad_sequence = True
                    break
            
            if bad_sequence:
                i += 1
                continue
            else:
                timestep = df.index[i + lookback - 1] # Index of the time step to forecast for
                feature_slice = df_slice.iloc[:lookback, :-1] # Exclude last column, which is the target
                target_slice = df_slice.iloc[lookback-1:, -1] # Only include last (target) column
                feature_tensor = torch.tensor(feature_slice.values.tolist())
                target_tensor = torch.tensor(target_slice.values.tolist())
                dataset[n] = (timestep, feature_tensor, target_tensor)
                n += 1
                i += 1

        self.ds = dataset # Dictionary of the form {index: (timestep, features, targets)}


    def __len__(self):
        # n = len(self.df.index.year.unique()) + 1 # Number of splits in the data + 1 -> num of segments
        # return len(self.df) - n*(self.lookback + self.num_forecast_steps - 2)
        return len(self.ds)
    
    def __getitem__(self, index):

        return self.ds[index][1], self.ds[index][2]

        # years = self.df.index.year.unique()
        # num_samples = 0
        # for year in years:
        #     start = datetime.datetime(year, 1, 1, 0)
        #     end = datetime.datetime(year, 12, 31, 23)
        #     df_subset = self.df.loc[start:end]
        #     lower_bound = num_samples
        #     num_samples += len(df_subset) - (self.lookback + self.num_forecast_steps - 2)
        #     upper_bound = num_samples
        #     if (index >= lower_bound) and (index < upper_bound):
                
        #         break
        # while True:
        #     df_slice = self.df.iloc[index:index+self.lookback+self.num_forecast_steps-2]
        #     num_unique_years = len(df_slice.index.year.unique())
        #     if num_unique_years < 2:
        #         break
        #     else:
        #         index += 1
        
        # feature_slice = df_slice.iloc[:self.lookback, :-1] # all columns but the last
        # target_slice = df_slice.iloc[-self.num_forecast_steps, -1] # tp is last column
        
        # feature_tensor = torch.from_numpy(feature_slice.values)
        # target_tensor = torch.from_numpy(target_slice.values)
        # return feature_tensor, target_tensor
