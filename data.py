import cfgrib
import itertools
import numpy as np
import os
import pandas as pd
import xarray as xr

# The data files are saved as .grib files in the folder ERA5_Data within the current working directory.
# There is a separate file for each year for the surface data
class DatasetLoader:
    def convert_dist_to_coords(dist: int, origin: tuple):
        pass

    def create_feature_dataframe(dataset: xr.Dataset, low_lat_bound: float, high_lat_bound: float, low_long_bound: float, high_long_bound: float, num_additional_points: int):
        """Creates a pandas dataframe from dataset; the resulting dataframe is saved to the system"""
        coords = []
        # Blacksburg coordinates are about (37.23, -80.42); add these first, data for this point will be present in all trials
        lat_points = dataset["latitude"].to_numpy()
        long_points = dataset["longitude"].to_numpy()
        bburg_lat_index = np.argmin(np.abs(lat_points - 37.23))
        bburg_long_index = np.argmin(np.abs(long_points + 80.42))
        coords.append((lat_points[bburg_lat_index],long_points[bburg_long_index]))

        num_additional_points = min(num_additional_points, len(lat_points) * len(long_points) - 1) # Prevent cases where total num points > num of points in dataset

        # Restrict the possible coordinate locations
        lat_points = lat_points[lat_points > low_lat_bound]
        lat_points = lat_points[lat_points < high_lat_bound]
        long_points = long_points[long_points > low_long_bound]
        long_points = long_points[long_points < high_long_bound]
        
        i = 0
        while i < num_additional_points:
            lat_index = np.random.randint(0, len(lat_points))
            long_index = np.random.randint(0, len(long_points))
            if (lat_points[lat_index], long_points[long_index]) not in coords:
                coords.append((lat_points[lat_index], long_points[long_index]))
                i += 1
            else:
                continue
        
        lat_coords = []
        long_coords = []
        for coord in coords:
            lat_coords.append(coord[0])
            long_coords.append(coord[1])
        refined_xarray = dataset.sel(dict(latitude=lat_coords, longitude=long_coords))
        df = refined_xarray.to_dataframe()
        df = df.drop(columns=["surface", "step", "number", "valid_time"])
        df = df.unstack([1, 2]) # Convert the MultiIndex of time, latitude, longtitude into a singular index of time; latitude and longtiude become column levels 1 and 2 respectively
        df.columns = df.columns.to_flat_index() # Merge all levels, the resulting column labels will be of the form (weather_var, latitude, longitude)

        # TODO: Save dataframe as csv file or something for easy access when training models

        # var_labels = list(df.columns.levels[0].values)
        # lat_labels = list(map(str, list(df.columns.levels[1].values)))
        # long_labels = list(map(str, list(df.columns.levels[2].values)))


    
    def join_datasets_by_year(datasets: dict):
        dataset_list = []
        for year in dict.keys():
            dataset_list.append(datasets[year])
        return xr.concat(datasets, dim="time")

    # def determine_dataset_variable_type(dataset: xr.Dataset):
    #     if "tp" in dataset.data_vars:
    #         # This is the dataset containing the total precipitation (accumulated variable); for this project, total precipitation is the only accumulated variable
    #         return "target"
    #     else:
    #         # All other instantaneous variables
    #         return "features"
        
    

dsl = DatasetLoader()
years = [2023, 2024]
surface_feature_datasets = {}
target_datasets = {}
for year in years:
    path = os.getcwd() + f"/ERA5_Data/{year}-full-surface-data.grib"
    surface_datasets = cfgrib.open_datasets(path) # first dataset contains the features, second contains target (tp, only accumulated var in this project)

    surface_feature_datasets[year] = surface_datasets[0]
    target_datasets[year] = surface_datasets[1]

surface_feature_dataset = dsl.join_datasets_by_year(surface_feature_datasets)
del surface_feature_datasets
target_dataset = dsl.join_datasets_by_year(target_datasets)
del target_datasets

    

# ds = xr.open_dataset(path, engine="cfgrib")
# df = ds.to_dataframe()
# print(df.head())