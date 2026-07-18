from dataclasses import dataclass, field
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

@dataclass(frozen=True)
class DataConfig:
    GRIB_DATA_FOLDER_NAME: str = "ERA5_Data"
    DATASET_METADATA_FILENAME: str = "metadata.csv"
    CSV_FOLDER_NAME: str = "CSV_Datasets"
    PROCESSED_CSV_FOLDER_NAME: str = "Processed_Datasets"
    NUM_VARIABLES: int = 13
    YEARS: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024])
    PRESSURES: list[int] = field(default_factory=lambda: [925, 850, 700, 500])
    

@dataclass(frozen=True)
class PreprocessingConfig:
    FEATURES_TO_DROP: list[str] = field(default_factory=lambda: [])
    VALID_PROPORTION: float = 0.10 # For validation set; 10% of the data from 2021 to 2023
    BUFFER_HOURS: int= 72 # Trimmed off both sides of the train/test-year boundary to reduce data leakage; see data_preprocessor.py
    TEST_YEAR: int = 2024

@dataclass(frozen=True)
class ExperimentConfig:
    ORIGIN: tuple = field(default_factory=lambda: (37.23, -80.42)) # Blacksburg, VA
    DISTS: list[int] = field(default_factory=lambda: [50, 75, 100, 125, 150])
    NUM_LOCATIONS: list[int] = field(default_factory=lambda: [4, 7, 10]) # The 1-location control has no distance/draw dependence; handled separately
    NUM_DRAWS: int = 5 # This is the number of times to sample data points for each distance bound and number of locations
    TRAINING_SEEDS: list[int] = field(default_factory=lambda: [1, 2, 3]) # This these are the seeds used to initialize training params; record avg performance over to training seeds to get more stable estimate accounting for optimization randomness

@dataclass(frozen=True)
class HyperparameterTestingConfig:
    # Following 4 exist in both MainConfig & HyperparameterTestingConfig
    MODEL_SAVE_FOLDER: str = "Hyperparameter_Testing_Models"
    RUNS_FOLDER: str = "Hyperparameter_Testing_Runs"
    RESULTS_PATH: str = "hyperparameter_testing_results.csv"
    RESULTS_COLUMNS: list[str] = field(default_factory=lambda: ["num_locations", "lookback", "num_hiddens", "weight_decay", "valid_F1", "F1"]) # Output file columns; can be renamed, not reordered or reshaped

    # These 3 are the possible hyperparams for testing; unique to HyperparameterTestingConfig
    LOOKBACKS: list[int] = field(default_factory=lambda: [5, 12, 24]) # Test all lookbacks with num_hiddens default = 64 and weight_decay default = 0 first; select best
    NUM_HIDDENS_OPTIONS: list[int] = field(default_factory=lambda: [32, 64, 128, 256]) # Then test varying numbers of num_hiddens on the best lookback, with weight_decay = 0; select best
    WEIGHT_DECAY_OPTIONS: list[float] = field(default_factory=lambda: [0, 1e-4, 1e-3]) # Finally, select best weight_decay for the best lookback and num_hiddens

@dataclass(frozen=True)
class MainConfig:
    # Following 4 exist in both MainConfig & HyperparameterTestingConfig
    MODEL_SAVE_FOLDER: str = "Models"
    RUNS_FOLDER: str = "Runs"
    RESULTS_PATH: str = "results.csv"
    RESULTS_COLUMNS: list[str] = field(default_factory=lambda: ["num_locations", "dist", "draw_num", "seed", "F1"]) # Output file columns; can be renamed, but not reordered or reshaped

    # These 3 are changed with hyperparameter selection; unique to MainConfig
    LOOKBACK: int = 5 # LOOKBACK - 1 hours before, since data for current timestep is included
    NUM_HIDDENS: int = 64
    WEIGHT_DECAY: float = 0

    # For baseline model
    BASELINE_RESULTS_PATH: str = "baseline_results.csv" # results cols are the same as for lstm models; baseline models aren't saved


@dataclass(frozen=True)
class FixedHyperparameterConfig:
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    MAX_EPOCHS: int = 100
    MIN_EPOCHS: int = 20
    GRADIENT_CLIP: float = 1.0
    EARLY_STOPPING_PATIENCE: int = 5
    WET_DAY_LOSS_WEIGHT: float = 4.0

def generate_file_label(num_locations: int, dist=50, draw_num=1):
    if num_locations == 1:
        return "1-site__control"
    else:
        return f"{num_locations}-sites__{dist}-miles__draw{draw_num}"


