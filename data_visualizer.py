import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import preprocessor

# df["t2m_diff"] = df["t2m@(37.11,-80.34)"] - df["t2m@(37.11,-80.34)"].shift(1)
# df["sp_diff"] = df["sp@(37.11,-80.34)"] - df["sp@(37.11,-80.34)"].shift(1)
# df.dropna(inplace=True)
# print(df.head())

def get_complete_dataset():
    path = os.getcwd() + "/CompleteDatasets/10-sites__100-miles.csv"
    df = pd.read_csv(path, index_col=0)
    # Convert dry days to 0 and wet days to 1
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 0 if x*1000 < 0.05 else 1)
    return df



def histogram(df, feature, ratio):
    plt.figure(figsize=(12,10))
    sns.histplot(data=df, x=feature, hue="tp@(37.11,-80.34)", multiple="layer", weights="weights")
    plt.title(f"Histogram of {feature} by dry/wet days; wet/dry day ratio = {ratio}")
    plt.show()

# def histograms(df, features):
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,6))
#     axes = axes.flatten()
#     for i, col in enumerate(features[0:8]):
#         sns.histplot(data=df, x=col, hue='tp@(37.11,-80.34)', multiple="stack")
#         axes[i].set_title(f'Distribution of {col}: Wet vs Dry')
#         axes[i].set_xlabel(f'{col}')
#         axes[i].set_ylabel(f'Count')
    
#     plt.tight_layout()
#     plt.show()



def kdeplot(features):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(features[0:4]):
        sns.kdeplot(data=df, x=col, hue='tp@(37.11,-80.34)', fill=True, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Distribution of {col}: Wet vs Dry')
        axes[i].set_xlabel(f'{col}')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.show()

features = ["q@925hPa@(37.11,-80.34)", "q@850hPa@(37.11,-80.34)", "q@700hPa@(37.11,-80.34)", "d2m@(37.11,-80.34)",
            "sp@(37.11,-80.34)", "t2m@(37.11,-80.34)", "vo@500hPa@(37.11,-80.34)", "ws10@(37.11,-80.34)",
            "ws@925hPa@(37.11,-80.34)", "ws@850hPa@(37.11,-80.34)", "ws@700hPa@(37.11,-80.34)", "t2m_diff", "sp_diff"]


def get_processed_datasets(folder="TestDatasets"):
    dp = preprocessor.DataPreprocessor(0.8, "CompleteDatasets", 10, 100, folder, ["ws10@", "ws@925", "ws@850", "ws@700"])
    dp.save_standardized_train_test_datasets()
    return pd.read_csv(os.getcwd() + "/TestDatasets/10-sites__100-miles__train.csv", index_col=0)

# df = get_processed_datasets()
df = pd.read_csv(os.getcwd() + "/SplitDatasets/10-sites__100-miles__train.csv", index_col=0)
wet_dry_ratio = len(df[df["tp@(37.11,-80.34)"] == 1]) / len(df[df["tp@(37.11,-80.34)"] == 0])
df["weights"] = df["tp@(37.11,-80.34)"].apply(lambda x: wet_dry_ratio if x==0 else 1)
histogram(df, features[6], wet_dry_ratio)