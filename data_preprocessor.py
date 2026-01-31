import math
import os
import pandas as pd

class ClassificationDataPreprocessor:
    def __init__(self, train_proportion: float, valid_proportion: float, block_proportion: float, num_blocks: int, original_folder: str, num_sites: int, dist: int, new_folder: str, features_to_drop):
        old_filename = f"{num_sites}-sites__{dist}-miles"
        path = os.getcwd() + "/" + original_folder + "/" + old_filename + ".csv"
        df = pd.read_csv(path, index_col=0) # Use the first column as the index

        # Drop any specified features
        cols_to_drop = []
        for col in list(df.columns):
            for f in features_to_drop:
                if col.startswith(f):
                    cols_to_drop.append(col)

        df.drop(cols_to_drop, axis=1, inplace=True)

        # Measure tp in mm
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1000*x)

        # Create blocks
        block_size = math.floor(block_proportion * len(df.index))
        delta = (len(df.index) - block_size) // num_blocks
        delta = max(1, delta) # delta cannot be 0
        blocks = []
        for i in range(num_blocks):
            start_index = i*delta
            end_index = start_index + block_size
            if end_index > len(df.index):
                print(f"Warning: end_index out of bounds; only {i+1} blocks out of {num_blocks} were able to be created")
                num_blocks = i+1
                break
            blocks.append(df.iloc[start_index:end_index, :])
        
        # Both means and stdevs are of the form {block_num: {col: mean or stdev, ...}, ...}
        self.means = {}
        self.stdevs = {}
        self.datasets = {} # Of the form {block_num: {train: dataset, valid: dataset, test: dataset}, ...}
        # In all dictionaries above, block_num is zero indexed, unlike when obtaining the means or stdevs from outside the class
        self.old_filename = old_filename
        self.new_folder = new_folder

        for i in range(num_blocks):
            train_df, valid_df, test_df = self._split_and_standardize(train_proportion, valid_proportion, i, blocks)
            self.datasets[i] = {
                "train": train_df,
                "valid": valid_df,
                "test": test_df
            }

    
    def _split_and_standardize(self, train_proportion, valid_proportion, block_num, blocks):
        df = blocks[block_num]
        valid_split_index = math.floor(train_proportion * len(df.index))
        test_split_index = math.floor((train_proportion + valid_proportion) * len(df.index))

        train_df = df.iloc[:valid_split_index, :]
        valid_df = df.iloc[valid_split_index:test_split_index, :]
        test_df = df.iloc[test_split_index:, :]

        # These store the means and stdevs for each column in train_df; used to standardize the data
        train_means = {}
        train_stdevs = {}
        for col in train_df.columns:
            train_means[col] = train_df[col].mean()
            train_stdevs[col] = train_df[col].std()
        self.means[block_num] = train_means
        self.stdevs[block_num] = train_stdevs

        # Standardize features; both train & test datasets have the same columns
        for col in train_df.columns[:-1]:
            train_df[col] = train_df[col].apply(lambda x: (x - train_means[col]) / train_stdevs[col])
            valid_df[col] = valid_df[col].apply(lambda x: (x - train_means[col]) / train_stdevs[col])
            test_df[col] = test_df[col].apply(lambda x: (x - train_means[col]) / train_stdevs[col])
        
        # Convert dry days to 0 and wet days to 1
        train_df.iloc[:, -1] = train_df.iloc[:, -1].apply(lambda x: 0 if x < 0.05 else 1)
        valid_df.iloc[:, -1] = valid_df.iloc[:, -1].apply(lambda x: 0 if x < 0.05 else 1)
        test_df.iloc[:, -1] = test_df.iloc[:, -1].apply(lambda x: 0 if x < 0.05 else 1)

        return train_df, valid_df, test_df
        

    def obtain_means(self, block_num):
        """block_num should be the number of the block or fold, with the first block being 1"""
        return self.means[block_num - 1]
    
    def obtain_stdevs(self, block_num):
        """block_num should be the number of the block or fold, with the first block being 1"""
        return self.stdevs[block_num - 1]
    
    
    def save_datasets(self):
        """Use this function to save the standardized train and test datasets;
        After being saved, the standardized datasets can be loaded by the WeatherDataset class"""

        general_folder_path = os.getcwd() + "/" + self.new_folder
        if not os.path.exists(general_folder_path):
            os.mkdir(general_folder_path)

        for block_num, block_datasets in self.datasets.items():
            block_folder_path = general_folder_path + f"/Block_{block_num + 1}" # Switch from internal 0-indexed block_num to user visible block_num indexed from 1
            if not os.path.exists(block_folder_path):
                os.mkdir(block_folder_path)
            
            for dataset_type, dataset in block_datasets.items():
                dataset.to_csv(block_folder_path + "/" + self.old_filename + f"__{dataset_type}.csv")
