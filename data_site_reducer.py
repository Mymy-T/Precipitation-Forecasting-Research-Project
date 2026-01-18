import os
import pandas as pd

class SiteReducer:
    def __init__(self, max_sites, dist, num_variables, processed_dataset_folder):
        self.num_variables = num_variables
        self.max_sites = max_sites
        self.dist = dist
        self.folder = processed_dataset_folder
    
    def save_reduced_site_dataset(self, num_sites):
        path = os.getcwd() + "/" + self.folder + f"/{self.max_sites}-sites__{self.dist}-miles__"
        train_df = pd.read_csv(path + "train.csv", index_col=0)
        valid_df = pd.read_csv(path + "valid.csv", index_col=0)
        test_df = pd.read_csv(path + "test.csv", index_col=0)
        end_col = num_sites*self.num_variables
        cols_to_remove = list(train_df.columns) # Both train and test datasets have the same columns
        cols_to_remove = cols_to_remove[end_col:-1]
        new_train_df = train_df.drop(cols_to_remove, axis=1)
        new_valid_df = valid_df.drop(cols_to_remove, axis=1)
        new_test_df = test_df.drop(cols_to_remove, axis=1)
        save_path = os.getcwd() + "/" + self.folder + f"/{num_sites}-sites__{self.dist}-miles__"
        new_train_df.to_csv(save_path + "train.csv")
        new_valid_df.to_csv(save_path + "valid.csv")
        new_test_df.to_csv(save_path + "test.csv")

    

sr = SiteReducer(10, 100, 7, "Split_Datasets")