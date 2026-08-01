import os
import pandas as pd
import variables

data_config = variables.DataConfig()
train_data_path = os.path.join(os.getcwd(), data_config.PROCESSED_CSV_FOLDER_NAME, f"{variables.generate_file_label(num_locations=1)}__train.csv")
train_df = pd.read_csv(train_data_path)
print(train_df["tp@(37.11,-80.34)"].value_counts())