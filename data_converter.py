import cfgrib
import math
import numpy as np
import os
import pandas as pd
import xarray as xr

# The data files are saved as .grib files in the folder ERA5_Data within the current working directory.
# There is a separate file for each year for the surface data
class DatasetLoader:

    def __init__(self, load_folder, save_folder, pressures):
        self.era5_folder_name = load_folder
        self.csv_folder_name = save_folder
        self.pressures = pressures

    def join_datasets_by_year(self, datasets: dict):
        dataset_list = []
        for year in datasets.keys():
            dataset_list.append(datasets[year])
        return xr.concat(dataset_list, dim="time")
    
    def get_pressure_datasets(self, year):
        datasets = []
        for pressure in self.pressures:
            path = os.getcwd() + f"/{self.era5_folder_name}/{pressure}hPa-{year}.grib"
            pressure_ds = cfgrib.open_dataset(path)
            pressure_ds = pressure_ds.drop_vars("isobaricInhPa") # Drop coordinate describing pressure level, since the pressure levels were downloaded on at a time for each pressure level and year
            if pressure != 500: # For this specific project, all pressures other than 500 hPa have q, u, v data while 500 hPa contains vo data
                pressure_ds = pressure_ds.rename({
                    "q": f"q@{pressure}hPa",
                    "u": f"u@{pressure}hPa",
                    "v": f"v@{pressure}hPa"
                })
            else:
                pressure_ds = pressure_ds.rename({
                    "vo": f"vo@{pressure}hPa"
                })
            datasets.append(pressure_ds)
        
        return xr.merge(datasets, compat="no_conflicts")
    
    def get_surface_datasets(self, year):
        """Returns surface features, target feature"""
        surface_path = os.getcwd() + f"/{self.era5_folder_name}/{year}-full-surface-data.grib"
        surface_datasets = cfgrib.open_datasets(surface_path) # first dataset contains the features, second contains target (tp, only accumulated var in this project)
        return surface_datasets[0], surface_datasets[1]
    
    def combine_feature_datasets(self, surface_dataset, pressure_dataset):
        return xr.merge([surface_dataset, pressure_dataset], compat="no_conflicts")


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
    

    def _combine_wind_components(self, df, lat, long):
        # Combine surface winds first
        wind_components = [f"u10@({lat},{long})", f"v10@({lat},{long})"]
        # indicies = []
        # for i in len(columns):
        #     if columns[i].startswith(wind_components):
        #         indicies.append(i)
        df[f"ws10@({lat},{long})"] = df.apply(lambda row: np.sqrt(row[wind_components[0]]**2 + row[wind_components[1]]**2), axis=1)
        df.drop(wind_components, axis=1, inplace=True)
        del wind_components
        # del indicies

        # Now, combine winds for pressure levels
        for pressure in self.pressures:
            if pressure != 500: # No wind speeds for 500 hPa pressure level
                wind_components = [f"u@{pressure}hPa@({lat},{long})", f"v@{pressure}hPa@({lat},{long})"]
                # indicies = []
                # for i in len(columns):
                #     if columns[i].startswith(wind_components):
                #         indicies.append(i)
                df[f"ws@{pressure}hPa@({lat},{long})"] = df.apply(lambda row: np.sqrt(row[wind_components[0]]**2 + row[wind_components[1]]**2), axis=1)
                df.drop(wind_components, axis=1, inplace=True)
        
        return df


        


    def save_dataframe(self, feature_dataset: xr.Dataset, target_dataset: xr.Dataset, num_points: int, dist: float, origin: tuple):
        """Creates a pandas dataframe from dataset; the resulting dataframe is saved as a csv;
            The default origin is set to the latitude and longitude coordinates of Blacksburg, VA;
            feature_dataset only contains ERA5 instantaneous variables, while target_dataset contains ERA5 total precipitation (accumulated variable)
        """
        coords = []
        # Add origin point data, data for this point will be present in all trials
        lat_points = feature_dataset["latitude"].to_numpy() # Both the feature and target datasets utilize the same ERA5 grid system
        long_points = feature_dataset["longitude"].to_numpy()
        origin_lat_index = np.argmin(np.abs(lat_points - origin[0]))
        origin_long_index = np.argmin(np.abs(long_points - origin[1]))
        origin_lat = lat_points[origin_lat_index]
        origin_long = long_points[origin_long_index]
        coords.append((origin_lat, origin_long))


        
        num_points = min(num_points, len(lat_points) * len(long_points)) # Prevent cases where num_points > num of points in dataset
        num_points = max(1, num_points) # Ensure num_points >= 1

        # Restrict the possible coordinate locations
        low_lat_bound, high_lat_bound, low_long_bound, high_long_bound = self._convert_dist_to_coords(dist=dist, origin=origin)
        lat_points = lat_points[lat_points > low_lat_bound]
        lat_points = lat_points[lat_points < high_lat_bound]
        long_points = long_points[long_points > low_long_bound]
        long_points = long_points[long_points < high_long_bound]
        
        i = 0
        while i < (num_points - 1): # Selected num_points - 1 extra points in addition to the origin point
            lat_index = np.random.randint(0, len(lat_points))
            long_index = np.random.randint(0, len(long_points))
            if (lat_points[lat_index], long_points[long_index]) not in coords:
                coords.append((lat_points[lat_index], long_points[long_index]))
                i += 1
            else:
                continue
        
        feature_df_subsets = []
        for coord in coords:
            # Each feature subset contains the feature data for a coordinate
            lat = coord[0]
            long = coord[1]
            # Slice the xarray by lat & long
            feature_xarray_subset = feature_dataset.sel(dict(latitude=lat, longitude=long))
            # Conversion to pandas dataframe
            feature_df_subset = feature_xarray_subset.to_dataframe()
            # Drop extra columns
            feature_df_subset = feature_df_subset.drop(columns=["surface", "step", "number", "valid_time", "latitude", "longitude"])
            # Format columns
            feature_new_cols = list(map(lambda x: f"{x}@({lat},{long})", list(feature_df_subset.columns))) # Each weather variable column name will be of the form 'weather_var@(latitude, longitude)'
            feature_df_subset.columns = feature_new_cols
            # Combine u & v wind components into a single wind speed feature (ws10@(lat, long), or ws@pressure@(lat, long))
            feature_df_subset = self._combine_wind_components(feature_df_subset, lat, long)

            feature_df_subsets.append(feature_df_subset)

        feature_df = pd.concat(feature_df_subsets, axis=1)

        # Get the target dataset
        target_xarray = target_dataset.sel(dict(latitude=origin_lat, longitude=origin_long))
        target_df = target_xarray.to_dataframe()
        # Drop extra columns
        target_df.drop(columns=["number", "surface", "latitude", "longitude"], inplace=True)
        # Format index & column
        target_df.set_index("valid_time", inplace=True)
        target_new_cols = list(map(lambda x: f"{x}@({origin_lat},{origin_long})", list(target_df.columns))) # Target variables @ the origin (location of interest)
        target_df.columns = target_new_cols
        # Remove any overlapping indicies resulting join_datasets_by_year(); note that a value of NaN is used for any date outside the range of a year specific dataset (e.g. total precipitation in 2023 will have NaN for the last values of 2022 and first values of 2024)
        target_df.dropna(inplace=True)
        # Shift all values up by 1, since tp measures the accumulated precipitation from valid_time - 1hr to valid_time
        full_year_dates = pd.date_range(start=target_df.index.min(), end=target_df.index.max(), freq="h")
        target_df = target_df.reindex(full_year_dates) # Prevent shifting from September into May
        target_df = target_df.shift(-1)
        # Match rows to the rows of feature_df (accumulated vars contain extra rows at the beginning and end)
        start = feature_df.index[0]
        end = feature_df_subset.index[-1]
        target_df = target_df[start:end]
        df = pd.concat([feature_df, target_df], axis=1)
        df.dropna(inplace=True)
        
        folder_path = os.getcwd() + "/" + self.csv_folder_name
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        save_path = folder_path + "/" + f"{num_points}-sites__{dist}-miles.csv"
        df.to_csv(save_path)

        
    
years = [2023, 2024]
pressures = [925, 850, 700, 500]
dsl = DatasetLoader(load_folder="ERA5_Data", save_folder="CompleteDatasets", pressures=pressures)

# Get datasets for each year
surface_feature_datasets = {}
pressure_datasets = {}
target_datasets = {}
for year in years:
    surface_feature_datasets[year], target_datasets[year] = dsl.get_surface_datasets(year)
    pressure_datasets[year] = dsl.get_pressure_datasets(year)

# Join by years
surface_feature_dataset = dsl.join_datasets_by_year(surface_feature_datasets)
del surface_feature_datasets
target_dataset = dsl.join_datasets_by_year(target_datasets)
del target_datasets
pressure_dataset = dsl.join_datasets_by_year(pressure_datasets)
del pressure_datasets

feature_dataset = dsl.combine_feature_datasets(surface_dataset=surface_feature_dataset, pressure_dataset=pressure_dataset)
del surface_feature_dataset
del pressure_dataset

for i in range(5):
    d = (i+1) * 100
    dsl.save_dataframe(feature_dataset=feature_dataset, target_dataset=target_dataset, num_points=10, dist=d, origin=(37.23, -80.42))
