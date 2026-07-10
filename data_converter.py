import cfgrib
import math
import numpy as np
import os
import pandas as pd
import xarray as xr
import variables

def main():
    data_config = variables.DataConfig()
    experiment_config = variables.ExperimentConfig()

    # For each num_locations and distance combination, draw num_draws site sets
    dsl = DatasetLoader(load_folder=data_config.GRIB_DATA_FOLDER_NAME, save_folder=data_config.CSV_FOLDER_NAME, pressures=data_config.PRESSURES)
    feature_dataset, target_dataset = _build_full_feature_target_datasets(dsl, data_config.YEARS)

    dataset_metadata_rows = []

    # The control is a single dataset containing only the origin point (no extra draws needed)
    control_coords = dsl.draw_coords(feature_dataset, num_points=1, dist=experiment_config.DISTS[0], origin=experiment_config.ORIGIN, rng=np.random.default_rng(0))
    control_file_label = variables.generate_file_label(num_locations=1)
    dsl.save_dataframe(feature_dataset, target_dataset, coords=control_coords, origin=experiment_config.ORIGIN, file_label=control_file_label)
    dataset_metadata_rows.append({"num_locations": 1, "dist": None, "draw_num": None, "seed": None, "coords": control_coords})

    for dist in experiment_config.DISTS:
        for num_locations in experiment_config.NUM_LOCATIONS:
            for draw_num in range(1, experiment_config.NUM_DRAWS + 1):
                # draw_num is a unique seed within every combination of num_locations and dist
                seed = dist * 10_000 + num_locations * 100 + draw_num
                rng = np.random.default_rng(seed)
                coords = dsl.draw_coords(feature_dataset, num_points=num_locations, dist=dist, origin=experiment_config.ORIGIN, rng=rng)
                file_label = variables.generate_file_label(num_locations=num_locations, dist=dist, draw_num=draw_num)
                dsl.save_dataframe(feature_dataset, target_dataset, coords=coords, origin=experiment_config.ORIGIN, file_label=file_label)
                dataset_metadata_rows.append({"num_locations": num_locations, "dist": dist, "draw_num": draw_num, "seed": seed, "coords": coords})
                print(f"Saved {file_label}")

    dataset_metadata_df = pd.DataFrame(dataset_metadata_rows)
    dataset_metadata_path = os.path.join(os.getcwd(), dsl.csv_folder_name, data_config.DATASET_METADATA_FILENAME)
    dataset_metadata_df.to_csv(dataset_metadata_path, index=False)
    print(f"Logged dataset metadata to {dataset_metadata_path}")

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
            path = os.path.join(os.getcwd(), self.era5_folder_name, f"{pressure}hPa-{year}.grib")
            pressure_ds = cfgrib.open_dataset(path)
            pressure_ds = pressure_ds.drop_vars("isobaricInhPa") # Drop coordinate describing pressure level, since the pressure levels were downloaded separately for each pressure level and year
            if pressure != 500: # For this specific project, all pressures other than 500 hPa have q, u, v data while 500 hPa contains vo data
                pressure_ds = pressure_ds.rename({
                    "q": f"q@{pressure}hPa",
                    "u": f"u@{pressure}hPa",
                    "v": f"v@{pressure}hPa"
                })
            else:
                # Accidentally downloaded u and v components for 500hPa for 2021, 2022; no effect on other years
                if "u" in pressure_ds:
                    pressure_ds = pressure_ds.drop_vars("u")
                if "v" in pressure_ds:
                    pressure_ds = pressure_ds.drop_vars("v")

                pressure_ds = pressure_ds.rename({
                    "vo": f"vo@{pressure}hPa"
                })
            datasets.append(pressure_ds)

        return xr.merge(datasets, compat="no_conflicts")

    def get_surface_datasets(self, year):
        """Returns surface features and the target variable"""
        surface_path = os.path.join(os.getcwd(), self.era5_folder_name, f"{year}-full-surface-data.grib")
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
        delta_long = (180 * dist) / (math.pi * earth_radius_miles * math.cos(origin_lat * math.pi/180) * math.sqrt(2))
        return origin_lat - delta_lat, origin_lat + delta_lat, origin_long - delta_long, origin_long + delta_long

    def _convert_coords_to_dist(self, coord: tuple, origin: tuple):
        origin_lat = origin[0]
        origin_long = origin[1]
        lat = coord[0]
        long = coord[1]
        earth_radius_miles = 3959
        x_dist = earth_radius_miles * math.cos(origin_lat * math.pi/180) * abs(long - origin_long) * math.pi/180
        y_dist = earth_radius_miles * abs(lat - origin_lat) * math.pi/180
        return math.sqrt(x_dist**2 + y_dist**2)

    def draw_coords(self, feature_dataset: xr.Dataset, num_points: int, dist: float, origin: tuple, rng: np.random.Generator):
        """
        - Independently draws num_points-1 random grid coordinates within dist miles of origin in additon to the origin point using rng
        - Returns the coordinates sorted by ascending distance to the origin
        - num_points >= 1
        """
        lat_points = feature_dataset["latitude"].to_numpy() # Both the feature and target datasets utilize the same ERA5 grid system
        long_points = feature_dataset["longitude"].to_numpy()
        origin_lat_index = np.argmin(np.abs(lat_points - origin[0]))
        origin_long_index = np.argmin(np.abs(long_points - origin[1]))
        origin_lat = lat_points[origin_lat_index]
        origin_long = long_points[origin_long_index]
        coords = [(origin_lat, origin_long)]

        num_points = min(num_points, len(lat_points) * len(long_points)) # Prevent cases where num_points > num of avaliable points in dataset
        num_points = max(1, num_points) # Ensure num_points >= 1

        # Restrict the possible coordinate locations
        low_lat_bound, high_lat_bound, low_long_bound, high_long_bound = self._convert_dist_to_coords(dist=dist, origin=origin)
        lat_candidates = lat_points[(lat_points > low_lat_bound) & (lat_points < high_lat_bound)]
        long_candidates = long_points[(long_points > low_long_bound) & (long_points < high_long_bound)]
        print(f"Num lat candidates for dist {dist} = {len(lat_candidates)}; num long candidates = {len(long_candidates)}")

        i = 0
        while i < (num_points - 1): # Select num_points - 1 extra points in addition to the origin point
            lat_index = rng.integers(0, len(lat_candidates))
            long_index = rng.integers(0, len(long_candidates))
            candidate = (lat_candidates[lat_index], long_candidates[long_index])
            if candidate not in coords:
                coords.append(candidate)
                i += 1

        # Sort the coordinates by pythagorean distance to the origin in ascending order (for readable column order)
        coords = sorted(coords, key=lambda x: self._convert_coords_to_dist(x, origin))
        return coords

    def save_dataframe(self, feature_dataset: xr.Dataset, target_dataset: xr.Dataset, coords: list, origin: tuple, file_label: str):
        """
        - Creates a pandas dataframe from xarray dataset for the given coordinates, and saves it as a csv
        named using file_label.
        - feature_dataset only contains ERA5 instantaneous variables, while target_dataset
        contains ERA5 total precipitation (accumulated variable).
        """
        lat_points = feature_dataset["latitude"].to_numpy()
        long_points = feature_dataset["longitude"].to_numpy()
        origin_lat_index = np.argmin(np.abs(lat_points - origin[0]))
        origin_long_index = np.argmin(np.abs(long_points - origin[1]))
        origin_lat = lat_points[origin_lat_index]
        origin_long = long_points[origin_long_index]

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
        target_df = target_df.reindex(full_year_dates) # Prevent shifting from September into May; reindex fills in index with continuous date range, and fills in new rows with NaN
        target_df = target_df.shift(-1)
        # Match rows to the rows of feature_df (accumulated vars contain extra rows at the beginning and end)
        start = feature_df.index[0]
        end = feature_df_subset.index[-1]
        target_df = target_df[start:end]
        df = pd.concat([feature_df, target_df], axis=1)
        df.dropna(inplace=True)

        folder_path = os.path.join(os.getcwd(), self.csv_folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        save_path = os.path.join(folder_path, f"{file_label}.csv")
        df.to_csv(save_path)


def _build_full_feature_target_datasets(dsl, years):
    surface_feature_datasets = {}
    pressure_datasets = {}
    target_datasets = {}
    for year in years:
        surface_feature_datasets[year], target_datasets[year] = dsl.get_surface_datasets(year)
        pressure_datasets[year] = dsl.get_pressure_datasets(year)

    surface_feature_dataset = dsl.join_datasets_by_year(surface_feature_datasets)
    del surface_feature_datasets
    target_dataset = dsl.join_datasets_by_year(target_datasets)
    del target_datasets
    pressure_dataset = dsl.join_datasets_by_year(pressure_datasets)
    del pressure_datasets

    feature_dataset = dsl.combine_feature_datasets(surface_dataset=surface_feature_dataset, pressure_dataset=pressure_dataset)
    del surface_feature_dataset
    del pressure_dataset
    return feature_dataset, target_dataset


if __name__ == "__main__":
    main()
