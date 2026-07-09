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
MODEL_SAVE_FOLDER = "Hyperparameter_Testing_Models"
RUNS_FOLDER = "Hyperparameter_Testing_Runs"
RESULTS_PATH = "hyperparameter_testing_results.csv"
# All saved hyperparameter testing models will have a file name of the form "{x}-sites__{lookback}__{num_hiddens}__{weight_decay}"

# Data Properties
NUM_VARIABLES = 13

# Hyperparams To Test
LOOKBACKS = [5, 12, 24] # Test all lookbacks with num_hiddens default = 64 and weight_decay default = 0 first; select best
NUM_HIDDENS_OPTIONS = [32, 64, 128, 256] # Then test varying numbers of num_hiddens on the best lookback, with weight_decay = 0; select best
WEIGHT_DECAY_OPTIONS = [0, 1e-4, 1e-3] # Finally, select best weight_decay for the best lookback and num_hiddens

# Training Hyperparams
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
MIN_EPOCHS = 20
GRADIENT_CLIP = 1
EARLY_STOPPING_PATIENCE = 5
WET_DAY_LOSS_WEIGHT = 4.0

# Output file columns
RESULTS_COLUMNS = ["num_locations", "lookback", "num_hiddens", "weight_decay", "F1"]

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
    
    test_hyperparams()


def train(lookback, num_hiddens, weight_decay, num_locations, data_file_label, run_name):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    train_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, file_label=data_file_label, lookback=lookback, dataset_type=DatasetType.train)
    valid_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, file_label=data_file_label, lookback=lookback, dataset_type=DatasetType.validation)
    test_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, file_label=data_file_label, lookback=lookback, dataset_type=DatasetType.test)
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    writer = SummaryWriter(log_dir=f"{RUNS_FOLDER}/{run_name}")
    model = Model(NUM_VARIABLES, num_locations, num_hiddens)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WET_DAY_LOSS_WEIGHT]))
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, weight_decay=weight_decay)
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
        return {}
    existing = pd.read_csv(path)
    keys = zip(existing["num_locations"], existing["lookback"], existing["num_hiddens"], existing["weight_decay"]) # Of the form [(lookback_1, num_hiddens_1, weight_decay_1,), ...]
    return dict(zip(keys, existing["F1"])) # Of the form [(lookback_1, num_hiddens_1, weight_decay_1): f1_score_1, ...] -> keys are tuples

def _append_result(row):
    path = os.path.join(os.getcwd(), RESULTS_PATH)
    file_exists = os.path.exists(path)
    with open(path, "a") as f:
        if not file_exists:
            f.write(",".join(RESULTS_COLUMNS) + "\n") # Header columns
        f.write(",".join(str(row[c]) for c in RESULTS_COLUMNS) + "\n") # Write a new row of data

def run_and_log(lookback, num_hiddens, weight_decay, num_locations, data_file_label, run_name):
    key = (num_locations, lookback, num_hiddens, weight_decay)
    completed = _load_completed_keys() # This returns a set of all the information for prior runs (specifically, lookback, num_hiddens, weight_decay); if another run happens to match the label data, it has already been done and should be be run again; this is useful in case the code breaks in the middle; it allows you to resume
    if key in completed:
        prior_f1 = completed[key]
        if pd.isna(prior_f1):
            raise ValueError(f"{key} exists in history but has no valid F1 score! See {os.path.join(os.getcwd(), RESULTS_PATH)}")
        return prior_f1
    test_f1 = train(lookback=lookback, num_hiddens=num_hiddens, weight_decay=weight_decay, num_locations=num_locations, data_file_label=data_file_label, run_name=run_name)
    row = {"num_locations": num_locations, "lookback": lookback, "num_hiddens": num_hiddens, "weight_decay": weight_decay, "F1": test_f1}
    print(f"{run_name} F1: {test_f1:.4f}")
    _append_result(row)
    return test_f1

def run_hyperparmeter_test(num_locations):
    if num_locations == 1:
        data_file_label = "1-sites__control"
    else:
        data_file_label = "10-sites__50-miles__draw1" # Any dataset with 10 locations will do
    
    lookback_results = {}
    for lookback in LOOKBACKS:
        test_f1_score = run_and_log(lookback=lookback, num_hiddens=64, weight_decay=0, num_locations=num_locations, data_file_label=data_file_label, run_name=f"{num_locations}_sites__{lookback}__64__0")
        lookback_results[lookback] = test_f1_score
    best_lookback = max(lookback_results, key=lookback_results.get)
    print(f"For {num_locations} location(s), best lookback appears to be {best_lookback}, with f1 score of {lookback_results[best_lookback]:.4f}")

    num_hiddens_results = {}
    for num_hiddens in NUM_HIDDENS_OPTIONS:
        test_f1_score = run_and_log(lookback=best_lookback, num_hiddens=num_hiddens, weight_decay=0, num_locations=num_locations, data_file_label=data_file_label, run_name=f"{num_locations}_sites__{best_lookback}__{num_hiddens}__0")
        num_hiddens_results[num_hiddens] = test_f1_score
    best_num_hiddens = max(num_hiddens_results, key=num_hiddens_results.get)
    print(f"For {num_locations} location(s), best num_hiddens appears to be {best_num_hiddens}, with f1 score of {num_hiddens_results[best_num_hiddens]:.4f}")

    weight_decay_results = {}
    for weight_decay in WEIGHT_DECAY_OPTIONS:
        test_f1_score = run_and_log(lookback=best_lookback, num_hiddens=best_num_hiddens, weight_decay=weight_decay, num_locations=num_locations, data_file_label=data_file_label, run_name=f"{num_locations}_sites__{best_lookback}__{best_num_hiddens}__{weight_decay}")
        weight_decay_results[weight_decay] = test_f1_score
    best_weight_decay = max(weight_decay_results, key=weight_decay_results.get)
    print(f"For {num_locations} location(s), best weight_decay appears to be {best_weight_decay}, with f1 score of {weight_decay_results[best_weight_decay]:.4f}")


def test_hyperparams():
    run_hyperparmeter_test(num_locations=1)
    run_hyperparmeter_test(num_locations=10)

    print("Brief hyperparameter tuning complete!")
    print(f"See {os.path.join(os.getcwd(), MODEL_SAVE_FOLDER)} for all evaluated models.")
    print(f"See {os.path.join(os.getcwd(), RESULTS_PATH)} for all performance data.")
    print(f"See {os.path.join(os.getcwd(), RUNS_FOLDER)} for training graphs to check for underfitting/overfitting.")


if __name__ == "__main__":
    main()