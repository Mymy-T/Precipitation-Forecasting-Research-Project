from dataset import PrecipitationClassificationDataset
from data_preprocessor import ClassificationDataPreprocessor
from data_site_reducer import SiteReducer
import model
import trainer
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

CSV_FOLDER_NAME = "CSV_Datasets"
PROCESSED_CSV_FOLDER_NAME = "Split_Datasets"
DISTS = [100, 200, 300, 400, 500]
NUM_SITES_LIST = [1, 4, 7, 10]

NUM_VARIABLES = 7
FEATURES_TO_DROP = ["ws10@", "ws@925", "ws@850", "ws@700"]
OVERSAMPLE_NUM = 0
LOOKBACK = 4
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32
NUM_HIDDENS = 256
LEARNING_RATE = 1e-3
EPOCHS = 25
GRADIENT_CLIP = 1
EARLY_STOPPING_PATIENCE = 3
WET_DAY_LOSS_WEIGHT = 4.0

def prepare_datasets():
    for dist in DISTS:
        max_sites = max(NUM_SITES_LIST)
        dp = ClassificationDataPreprocessor(train_proportion=TRAIN_TEST_SPLIT, original_folder=CSV_FOLDER_NAME, num_sites=max_sites, dist=dist, new_folder=PROCESSED_CSV_FOLDER_NAME, features_to_drop=FEATURES_TO_DROP)
        dp.save_standardized_train_test_datasets()

        for num_sites in NUM_SITES_LIST:
            if num_sites != max_sites:
                sr = SiteReducer(max_sites=max_sites, dist=dist, num_variables=NUM_VARIABLES, processed_dataset_folder=PROCESSED_CSV_FOLDER_NAME)
                sr.save_reduced_site_dataset(num_sites=num_sites)


def train(num_locations, distance, experiment_name):
    train_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, train=True, oversample_num=OVERSAMPLE_NUM)
    test_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, train=False, oversample_num=0)
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
    model = model.Model(NUM_VARIABLES, num_locations, NUM_HIDDENS, BATCH_SIZE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WET_DAY_LOSS_WEIGHT]))
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    trainer = trainer.Trainer((train_dataloader, test_dataloader), model, loss_fn, optimizer, GRADIENT_CLIP)

    best_fscore = 0 # For early stopping
    for i in range(EPOCHS):
        epoch = i+1
        print(f"EPOCH: {epoch} \n------------------------------------------")
        train_loss = trainer.train_loop()
        test_loss, test_accuracy, test_f1 = trainer.test_loop()
        baseline_loss, baseline_accuracy, baseline_f1 = trainer.baseline_loop()
        writer.add_scalars(main_tag="Loss",
            tag_scalar_dict={
                "Train":train_loss,
                "Test":test_loss,
                "Baseline":baseline_loss
            },
            global_step=epoch
        )
        writer.add_scalars(main_tag="Accuracy",
            tag_scalar_dict={
                "Test":test_accuracy,
                "Basline":baseline_accuracy,
            },
            global_step=epoch
        )
        writer.add_scalars(main_tag="f1",
            tag_scalar_dict={
                "Test":test_f1,
                "Basline":baseline_f1,
            },
            global_step=epoch
        )
        writer.flush()
        print(f"Epoch {epoch} complete! \n--------------------------------")

    print("Training complete!")

# Run prepare_datasets once for every trial
# prepare_datasets()

experiment_name = "fscore2"
num_locations = 1
distance = 100
train(num_locations=num_locations, distance=distance, experiment_name=experiment_name)