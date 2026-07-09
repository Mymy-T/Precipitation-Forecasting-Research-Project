import os
import random
import numpy as np
import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import PrecipitationClassificationDataset, DatasetType
from model import Model
from trainer import Trainer


# Important Paths
PROCESSED_CSV_FOLDER_NAME = "Processed_Datasets"
MODEL_SAVE_FOLDER = "Models"
RUNS_FOLDER = "Runs"
RESULTS_PATH = "results.csv"

# Experimental Settings
DISTS = [50, 75, 100, 125, 150]
NUM_LOCATIONS = [4, 7, 10] # The 1-location control has no distance/draw dependence; handled separately
NUM_DRAWS = 5 # This is the number of times to sample data points for each distance bound and number of locations
TRAINING_SEEDS = [1, 2, 3] # This these are the seeds used to initialize training params; record avg performance over to training seeds to get more stable estimate accounting for optimization randomness

# Data Properties
NUM_VARIABLES = 13

# Training Hyperparams
LOOKBACK = 5 # LOOKBACK - 1 hours before, since data for current timestep is included
BATCH_SIZE = 32
NUM_HIDDENS = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
MIN_EPOCHS = 20
GRADIENT_CLIP = 1
EARLY_STOPPING_PATIENCE = 5
WET_DAY_LOSS_WEIGHT = 4.0

# Output file columns
RESULTS_COLUMNS = ["num_locations", "dist", "draw_id", "seed", "F1"]

def main():
    # Ensure the use runs this file from the right directory
    terminal_dir = os.path.abspath(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if terminal_dir != script_dir:
        sys.exit(
            f"ERROR: Script executed from the wrong folder.\n"
            f"Current working directory: {terminal_dir}\n"
            f"Expected working directory: {script_dir}"
        )
    
    run_all_combinations()


def train(file_label, num_locations, train_seed, run_name):
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    train_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, file_label=file_label, lookback=LOOKBACK, dataset_type=DatasetType.train)
    valid_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, file_label=file_label, lookback=LOOKBACK, dataset_type=DatasetType.validation)
    test_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, file_label=file_label, lookback=LOOKBACK, dataset_type=DatasetType.test)
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    writer = SummaryWriter(log_dir=f"{RUNS_FOLDER}/{run_name}")
    model = Model(NUM_VARIABLES, num_locations, NUM_HIDDENS)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WET_DAY_LOSS_WEIGHT]))
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    trainer = Trainer((train_dataloader, valid_dataloader, test_dataloader), model, loss_fn, optimizer, GRADIENT_CLIP, printing_metrics=False)

    model_folder_path = os.path.join(os.getcwd(), MODEL_SAVE_FOLDER)
    if not os.path.exists(model_folder_path):
        os.mkdir(model_folder_path)
    model_save_path = os.path.join(model_folder_path, f"model__{run_name}")

    best_valid_fscore = 0 # For early stopping using the validation set
    counter = 0 # Counts number of times valid_fscore didn't improve from last epoch

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy, train_f1 = trainer.train_loop()
        valid_loss, valid_accuracy, valid_f1 = trainer.test_loop(valid=True)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Train": train_loss, "Valid": valid_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"Train": train_accuracy, "Valid": valid_accuracy}, global_step=epoch)
        writer.add_scalars(main_tag="F1 Score", tag_scalar_dict={"Train": train_f1, "Valid": valid_f1}, global_step=epoch)
        writer.flush()

        if valid_f1 > best_valid_fscore:
            best_valid_fscore = valid_f1
            counter = 0
            torch.save(model.state_dict(), model_save_path) # Saves the best version; all other later saves override the previous model save
        else:
            if epoch <= MIN_EPOCHS:
                continue
            counter += 1

        if counter >= EARLY_STOPPING_PATIENCE:
            print(f"[{run_name}] stopped early, on epoch {epoch} of {MAX_EPOCHS}")
            break

    model.load_state_dict(torch.load(model_save_path, weights_only=True)) # Load the best model
    _, test_accuracy, test_f1 = trainer.test_loop(valid=False)
    writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"Test": test_accuracy}, global_step=1)
    writer.add_scalars(main_tag="F1 Score", tag_scalar_dict={"Test": test_f1}, global_step=1)
    writer.flush()
    return test_f1

def _load_completed_keys():
    path = os.path.join(os.getcwd(), RESULTS_PATH)
    if not os.path.exists(path):
        return set()
    existing = pd.read_csv(path)
    return set(
        zip(existing["num_locations"], existing["dist"], existing["draw_id"], existing["seed"]) # Returns a set of the form [(num_locations_1, dist_1, draw_id_1, seed_1), ...]
    )

def _append_result(row):
    path = os.getcwd() + "/" + RESULTS_PATH
    file_exists = os.path.exists(path)
    with open(path, "a") as f:
        if not file_exists:
            f.write(",".join(RESULTS_COLUMNS) + "\n") # Header columns
        f.write(",".join(str(row[c]) for c in RESULTS_COLUMNS) + "\n") # Write a new row of data

def run_and_log(num_locations, dist, draw_id, train_seed, file_label, run_name):
    key = (num_locations, dist, draw_id, train_seed)
    completed = _load_completed_keys() # This returns a set of all the information for prior runs (specifically, num_locations, dist, draw_id, train_seed); if another run happens to match the label data, it has already been done and should be be run again; this is useful in case the code breaks in the middle; it allows you to resume
    if key in completed:
        print(f"{run_name} has already been completed. Skipping...")
        return
    test_f1 = train(file_label=file_label, num_locations=num_locations, train_seed=train_seed, run_name=run_name)
    row = {"num_locations": num_locations, "dist": dist, "draw_id": draw_id, "seed": train_seed, "F1": test_f1}
    print(f"{run_name} F1: {test_f1:.4f}")
    _append_result(row)


def run_all_combinations():

    for train_seed in TRAINING_SEEDS: # Run for control (n_locations = 1)
        run_and_log(num_locations=1, dist=0, draw_id=1, train_seed=train_seed, file_label="1-sites__control", run_name=f"control__seed{train_seed}")

    for dist in DISTS: # Run for all other combinations
        for num_locations in NUM_LOCATIONS:
            for draw_id in range(1, NUM_DRAWS + 1):
                file_label = f"{num_locations}-sites__{dist}-miles__draw{draw_id}"
                for train_seed in TRAINING_SEEDS:
                    run_name = f"{file_label}__seed{train_seed}"
                    run_and_log(num_locations, dist, draw_id, train_seed, file_label, run_name)


if __name__ == "__main__":
    main()