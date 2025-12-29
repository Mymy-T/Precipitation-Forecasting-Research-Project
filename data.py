import cfgrib
import itertools
import math
import numpy as np
import os
import pandas as pd
import xarray as xr

# The data files are saved as .grib files in the folder ERA5_Data within the current working directory.
# There is a separate file for each year for the surface data
class DatasetLoader:
    def _convert_dist_to_coords(self, dist: float, origin: tuple):
        """The specified distance should be in miles, and the origin should be a tuple of the form (latitude, longitude);
            Returns 4 variables: low_lat_bound, high_lat_bound, low_long_bound, high_long_bound
        """
        origin_lat = origin[0]
        origin_long = origin[1]
        earth_radius_miles = 3959
        delta_lat = (180 * dist) / (math.pi * earth_radius_miles * math.sqrt(2))
        delta_long = (180 * dist) / (math.pi * earth_radius_miles * math.cos(origin_lat) * math.sqrt(2))
        return origin_lat - delta_lat, origin_lat + delta_lat, origin_long - delta_long, origin_long + delta_long


    def save_feature_dataframe(self, dataset: xr.Dataset, num_additional_points: int, dist: float, origin = (37.23, -80.42), folder_name = "CSV_Datasets"):
        """Creates a pandas dataframe from dataset; the resulting dataframe is saved as a csv;
            The default origin is set to the latitude and longitude coordinates of Blacksburg, VA
        """
        coords = []
        # Blacksburg coordinates are about (37.23, -80.42); add these first, data for this point will be present in all trials
        lat_points = dataset["latitude"].to_numpy()
        long_points = dataset["longitude"].to_numpy()
        origin_lat_index = np.argmin(np.abs(lat_points - origin[0]))
        origin_long_index = np.argmin(np.abs(long_points - origin[1]))
        coords.append((lat_points[origin_lat_index],long_points[origin_long_index]))



        num_additional_points = min(num_additional_points, len(lat_points) * len(long_points) - 1) # Prevent cases where total num points > num of points in dataset

        # Restrict the possible coordinate locations
        low_lat_bound, high_lat_bound, low_long_bound, high_long_bound = self._convert_dist_to_coords(dist=dist, origin=origin)
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
        
        # lat_coords = []
        # long_coords = []
        df_subsets = []
        for coord in coords:
            # lat_coords.append(coord[0])
            # long_coords.append(coord[1])
            xarray_subset = dataset.sel(dict(latitude=coord[0], longitude=coord[1]))
            df_subset = xarray_subset.to_dataframe()
            lat = df_subset["latitude"].iloc[0] # Get first element of latitude col, all vals are the same for each subset
            long = df_subset["longitude"].iloc[0] # Get first element of longitude col, all vals are the same for each subset
            df_subset = df_subset.drop(columns=["surface", "step", "number", "valid_time", "latitude", "longitude"])
            new_cols = list(map(lambda x: f"{x}@({lat},{long})", list(df_subset.columns))) # Each weather variable column name will be of the form 'weather_var@(latitude, longitude)'
            df_subset.columns = new_cols
            df_subsets.append(df_subset)
        
        df = pd.concat(df_subsets, axis=1)
        folder_path = os.getcwd() + "/" + folder_name
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        save_path = folder_path + "/" + f"features--{num_additional_points + 1}_sites--{dist}_miles.csv"
        df.to_csv(save_path)

            
        # refined_xarray = dataset.sel(dict(latitude=lat_coords, longitude=long_coords))
        # df = refined_xarray.to_dataframe()
        # df = df.drop(columns=["surface", "step", "number", "valid_time"])
        # df = df.unstack([1, 2]) # Convert the MultiIndex of time, latitude, longtitude into a singular index of time; latitude and longtiude become column levels 1 and 2 respectively
        #df.columns = df.columns.to_flat_index() # Merge all levels, the resulting column labels will be of the form (weather_var, latitude, longitude)

        # var_labels = list(df.columns.levels[0].values)
        # lat_labels = list(map(str, list(df.columns.levels[1].values)))
        # long_labels = list(map(str, list(df.columns.levels[2].values)))


    
    def join_datasets_by_year(self, datasets):
        dataset_list = []
        for year in datasets.keys():
            dataset_list.append(datasets[year])
        return xr.concat(dataset_list, dim="time")
        
    

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

for i in range(5):
    d = (i+1) * 100
    dsl.save_feature_dataframe(dataset=surface_feature_dataset, num_additional_points=15, dist=d)
