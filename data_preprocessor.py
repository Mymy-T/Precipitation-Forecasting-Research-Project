import math
import os
import pandas as pd

# Paths
CSV_FOLDER_NAME = "CSV_Datasets"
PROCESSED_CSV_FOLDER_NAME = "Processed_Datasets"

# Data Processing Settings
FEATURES_TO_DROP = []
VALID_PROPORTION = 0.10 # For validation set; 10% of the data from 2021 to 2023
BUFFER_HOURS = 72 # Trimmed off both sides of the train/test-year boundary to reduce data leakage; see data_preprocessor.py
TEST_YEAR = 2024

# Experimental Settings
DISTS = [50, 75, 100, 125, 150]
NUM_LOCATIONS = [4, 7, 10] # The 1-location control has no distance/draw dependence; handled separately
NUM_DRAWS = 5 # This is the number of times to sample data points for each distance bound and number of locations

class ClassificationDataPreprocessor:
    """
    Loads a single raw (features + target) CSV for one location-draw and creates
    chronological train / valid / test split:
      - test = the full held-out calendar year (test_year, e.g. 2024); whole year used to improve assessment of generalization
      - train / valid = the remaining years, split chronologically by valid_proportion.
      - buffer_hours rows are trimmed off both sides of the train-vs-test boundary due to potential similarlity between the tail of the validation dataset and the head of the test dataset -> reduce data leakage
    """
    def __init__(self, valid_proportion: float, buffer_hours: int, test_year: int, original_folder: str, file_label: str, new_folder: str, features_to_drop):
        path = os.path.join(os.getcwd(), original_folder, f"{file_label}.csv")
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)

        # Drop features (if specified)
        cols_to_drop = [col for col in df.columns for f in features_to_drop if col.startswith(f)]
        df.drop(cols_to_drop, axis=1, inplace=True)

        # Measure tp in mm
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1000 * x)

        is_test_year = df.index.year == test_year
        trainval_df = df[~is_test_year]
        test_df = df[is_test_year]

        if buffer_hours > 0:
            trainval_df = trainval_df.iloc[:-buffer_hours]
            test_df = test_df.iloc[buffer_hours:]

        valid_split_index = math.floor((1 - valid_proportion) * len(trainval_df.index))
        train_df = trainval_df.iloc[:valid_split_index, :].copy()
        valid_df = trainval_df.iloc[valid_split_index:, :].copy()
        test_df = test_df.copy()

        # These store the means and stdevs for each column in train_df; used to standardize the data
        train_means = {}
        train_stdevs = {}
        for col in train_df.columns[:-1]:
            train_means[col] = train_df[col].mean()
            train_stdevs[col] = train_df[col].std()
        self.means = train_means
        self.stdevs = train_stdevs

        # Standardize features using train-only stats; all three splits share the same columns
        for col in train_df.columns[:-1]:
            train_df[col] = (train_df[col] - train_means[col]) / train_stdevs[col]
            valid_df[col] = (valid_df[col] - train_means[col]) / train_stdevs[col]
            test_df[col] = (test_df[col] - train_means[col]) / train_stdevs[col]

        # Convert dry hours to 0 and wet hours to 1; wet threshold is 0.05 mm of hourly precipitation
        for df_split in (train_df, valid_df, test_df):
            df_split.iloc[:, -1] = df_split.iloc[:, -1].apply(lambda x: 0 if x < 0.05 else 1)

        self.file_label = file_label
        self.new_folder = new_folder
        self.datasets = {"train": train_df, "valid": valid_df, "test": test_df}

    def save_datasets(self):
        """Saves the standardized train/valid/test splits; loaded later by PrecipitationClassificationDataset."""
        folder_path = os.path.join(os.getcwd(), self.new_folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        for dataset_type, dataset in self.datasets.items():
            filename = f"{self.file_label}__{dataset_type}.csv"
            dataset.to_csv(os.path.join(folder_path, filename))


def prepare_datasets(dists, nums_locations, nums_draws):

    def get_all_file_labels():
        all_file_labels = []
        all_file_labels.append("1-sites__control")
        for dist in dists:
            for num_locations in nums_locations:
                for draw_id in range(1, nums_draws + 1):
                    all_file_labels.append(f"{num_locations}-sites__{dist}-miles__draw{draw_id}")
        return all_file_labels

    all_file_labels = get_all_file_labels() 
    for file_label in all_file_labels:
        dp = ClassificationDataPreprocessor(
            valid_proportion=VALID_PROPORTION,
            buffer_hours=BUFFER_HOURS,
            test_year=TEST_YEAR,
            original_folder=CSV_FOLDER_NAME,
            file_label=file_label,
            new_folder=PROCESSED_CSV_FOLDER_NAME,
            features_to_drop=FEATURES_TO_DROP,
        )
        dp.save_datasets()


if __name__ == "__main__":
    prepare_datasets(dists=DISTS, nums_locations=NUM_LOCATIONS, nums_draws=NUM_DRAWS)