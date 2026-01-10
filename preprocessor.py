import math
import os
import pandas as pd
import torch

class DataPreprocessor:
    def __init__(self, train_proportion: float, original_folder: str, num_sites: int, dist: int, new_folder: str, features_to_drop):
        old_filename = f"{num_sites}-sites__{dist}-miles"
        path = os.getcwd() + "/" + original_folder + "/" + old_filename + ".csv"
        df = pd.read_csv(path, index_col=0) # Use the first column as the index
        split_index = math.floor(train_proportion * len(df.index))

        # Drop some features
        # columns_to_drop = list(df.columns)[0:2] + list(df.columns)[6:-1]
        cols_to_drop = []
        for col in list(df.columns):
            for f in features_to_drop:
                if col.startswith(f):
                    cols_to_drop.append(col)

        df.drop(cols_to_drop, axis=1, inplace=True)

        # For access in other funcs
        self.train_df = df.iloc[:split_index, :]
        self.test_df = df.iloc[split_index:, :]
        self.old_filename = old_filename
        self.new_folder = new_folder

        # Measure tp in one tenths of mm (less precision needed)
        self.train_df.iloc[:, -1] = self.train_df.iloc[:, -1].apply(lambda x: 10000*x)
        self.test_df.iloc[:, -1] = self.test_df.iloc[:, -1].apply(lambda x: 10000*x)

        # These store the means and stdevs for each column in train_df; used to standardize the data
        self.train_means = {}
        self.train_stdevs = {}
        for col in self.train_df.columns:
            self.train_means[col] = self.train_df[col].mean()
            self.train_stdevs[col] = self.train_df[col].std()
        
    
    def save_train_test_datasets(self):
        """Use this function if you want to save the train and test datasets before they're preprocessed"""
        folder_path = os.getcwd() + "/" + self.new_folder + "_Not_Standardized"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        self.train_df.to_csv(folder_path + "/" + self.old_filename + "__train" + ".csv")
        self.test_df.to_csv(folder_path + "/" +self.old_filename + "__test" + ".csv")
    
    def obtain_train_means(self):
        return self.train_means
    
    def obtain_train_stds(self):
        return self.train_stdevs
    
    
    def save_standardized_train_test_datasets(self):
        """Use this function to save the standardized train and test datasets;
        After being saved, the standardized datasets can be loaded by the WeatherDataset class"""
        train_df = self.train_df
        test_df = self.test_df

        # Both train & test datasets have the same columns

        # Standardize features
        for col in train_df.columns[:-1]:
            train_df[col] = train_df[col].apply(lambda x: (x - self.train_means[col]) / self.train_stdevs[col])
            test_df[col] = test_df[col].apply(lambda x: (x - self.train_means[col]) / self.train_stdevs[col])
        
        # Min max scaling for targets
        # train_df.iloc[:, -1] = train_df.iloc[:, -1].apply(lambda x: (x-min(train_df.iloc[:, -1])) / (max(train_df.iloc[:, -1]) - min(train_df.iloc[:, -1])))
        # test_df.iloc[:, -1] = test_df.iloc[:, -1].apply(lambda x: (x-min(train_df.iloc[:, -1])) / (max(train_df.iloc[:, -1]) - min(train_df.iloc[:, -1])))

        # Log scaling for targets
        # train_df.iloc[:, -1] = train_df.iloc[:, -1].apply(lambda x: math.log(x + 1))
        # test_df.iloc[:, -1] = test_df.iloc[:, -1].apply(lambda x: math.log(x+1))

        # Convert dry days to 0 and wet days to 1
        train_df.iloc[:, -1] = train_df.iloc[:, -1].apply(lambda x: 0 if x < 0.05 else 1)
        test_df.iloc[:, -1] = test_df.iloc[:, -1].apply(lambda x: 0 if x < 0.05 else 1)

        folder_path = os.getcwd() + "/" + self.new_folder
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        train_df.to_csv(folder_path + "/" + self.old_filename + "__train" + ".csv")
        test_df.to_csv(folder_path + "/" +self.old_filename + "__test" + ".csv")


# dp = DataPreprocessor(0.8, "Datasets", 16, 100, "Test")