from dataset import PrecipitationClassificationDataset, DatasetType
from data_preprocessor import ClassificationDataPreprocessor
from data_site_reducer import SiteReducer
from model import Model
import os
from trainer import Trainer
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

CSV_FOLDER_NAME = "Large_CSV_Datasets"
PROCESSED_CSV_FOLDER_NAME = "Processed_Datasets"
MODEL_SAVE_FOLDER = "Models"
DISTS = [50, 75, 100, 125, 150]
NUM_SITES_LIST = [1, 2, 4, 6, 8, 10]

NUM_VARIABLES = 7
FEATURES_TO_DROP = ["ws10@", "ws@925", "ws@850", "ws@700"]
OVERSAMPLE_NUM = 0
LOOKBACK = 5 # LOOKBACK - 1 hours before, since data for current timestep is included
NUM_BLOCKS = 4
BLOCK_PROPORTION = 0.5
TRAIN_PROPORTION = 0.7
VALID_PROPORTION = 0.15
# TEST_PROPROTION is just 1 - TRAIN_PROPORTION - VALID_PROPORTION, implicity provided
BATCH_SIZE = 32
NUM_HIDDENS = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
MIN_EPOCHS = 20
GRADIENT_CLIP = 1
EARLY_STOPPING_PATIENCE = 5
WET_DAY_LOSS_WEIGHT = 4.0

def prepare_datasets():
    for dist in DISTS:
        max_sites = max(NUM_SITES_LIST)
        dp = ClassificationDataPreprocessor(train_proportion=TRAIN_PROPORTION, valid_proportion=VALID_PROPORTION, block_proportion=BLOCK_PROPORTION, num_blocks=NUM_BLOCKS, original_folder=CSV_FOLDER_NAME, num_sites=max_sites, dist=dist, new_folder=PROCESSED_CSV_FOLDER_NAME, features_to_drop=FEATURES_TO_DROP)
        dp.save_datasets()

        for num_sites in NUM_SITES_LIST:
            if num_sites != max_sites:
                sr = SiteReducer(max_sites=max_sites, dist=dist, num_variables=NUM_VARIABLES, num_blocks=NUM_BLOCKS, processed_dataset_folder=PROCESSED_CSV_FOLDER_NAME)
                sr.save_reduced_site_datasets(num_sites=num_sites)


def train(num_locations, distance, experiment_name, block_num):
    """Block num should be indexed from 1 when passed in"""
    train_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, dataset_type=DatasetType.train, block_num=block_num, oversample_num=OVERSAMPLE_NUM)
    valid_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, dataset_type=DatasetType.validation, block_num=block_num, oversample_num=0)
    test_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, dataset_type=DatasetType.test, block_num=block_num, oversample_num=0)
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}__block-{block_num}")
    model = Model(NUM_VARIABLES, num_locations, NUM_HIDDENS, BATCH_SIZE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WET_DAY_LOSS_WEIGHT]))
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    trainer = Trainer((train_dataloader, valid_dataloader, test_dataloader), model, loss_fn, optimizer, GRADIENT_CLIP, printing_metrics=False)

    model_folder_path = os.getcwd() + "/" + MODEL_SAVE_FOLDER
    if not os.path.exists(model_folder_path):
        os.mkdir(model_folder_path)
    model_save_path = model_folder_path + f"/model__{num_locations}-sites-{distance}-miles__block-{block_num}"

    best_valid_fscore = 0 # For early stopping using the validation set
    counter = 0

    for i in range(MAX_EPOCHS):
        epoch = i+1
        print(f"EPOCH: {epoch} \n------------------------------------------")
        train_loss, train_accuracy, train_f1 = trainer.train_loop()
        valid_loss, valid_accuracy, valid_f1 = trainer.test_loop(valid=True)
        _, baseline_accuracy, baseline_f1 = trainer.baseline_loop()
        random_accuracy, random_f1 = trainer.random_guessing_loop()
        writer.add_scalars(main_tag="Loss",
            tag_scalar_dict={
                "Train":train_loss,
                "Valid":valid_loss
            },
            global_step=epoch
        )
        writer.add_scalars(main_tag="Accuracy",
            tag_scalar_dict={
                "Train":train_accuracy,
                "Valid":valid_accuracy,
                "Baseline":baseline_accuracy,
                "Random":random_accuracy
            },
            global_step=epoch
        )
        writer.add_scalars(main_tag="F1 Score",
            tag_scalar_dict={
                "Train":train_f1,
                "Valid":valid_f1,
                "Baseline":baseline_f1,
                "Random":random_f1
            },
            global_step=epoch
        )
        writer.flush()

        print(f"Epoch {epoch} complete! \n--------------------------------")
        # Early stopping stuff
        if valid_f1 > best_valid_fscore:
            best_valid_fscore = valid_f1
            counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            if epoch <= MIN_EPOCHS:
                continue
            else:
                counter += 1
                print(f"{counter}/{EARLY_STOPPING_PATIENCE} early stops")
        
        if counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stop; epoch {epoch}/{MAX_EPOCHS}")
            break

    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    _, test_accuracy, test_f1 = trainer.test_loop(valid=False)
    writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"Test":test_accuracy}, global_step=1)
    writer.add_scalars(main_tag="F1 Score", tag_scalar_dict={"Test":test_f1}, global_step=1)
    writer.flush()
    print(f"Test F1 Score, Block {block_num}: {test_f1: .4f}")
    return test_f1, epoch - EARLY_STOPPING_PATIENCE

def train_all_blocks(num_locations, distance, experiment_name):
    f1_scores = []
    epochs = []
    for i in range(NUM_BLOCKS):
        test_f1, epoch = train(num_locations, distance, experiment_name, i+1)
        f1_scores.append(test_f1)
        epochs.append(epoch)
    
    # Print final test scores; record in a spreadsheet 
    for i in range(NUM_BLOCKS):
        print(f"F1 (block {i+1}): {f1_scores[i]: .4f}; epoch: {epochs[i]}")

# # Run prepare_datasets before training
# prepare_datasets()

trial_num = 10
num_locations = 1
distance = 50
experiment_name = f"trial-{trial_num}__{num_locations}-sites__{distance}-miles"
train_all_blocks(num_locations=num_locations, distance=distance, experiment_name=experiment_name)