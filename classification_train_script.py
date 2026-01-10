from classification_dataset import PrecipitationClassificationDataset
from classification_preprocessor import ClassificationDataPreprocessor
import classification_model
import classification_trainer
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

CSV_FOLDER_NAME = "CompleteDatasets"
PROCESSED_CSV_FOLDER_NAME = "Split_Datasets"
EXPERIMENT_NAME = "classify4"

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

num_locations = 10
distance = 100

dp = ClassificationDataPreprocessor(train_proportion=TRAIN_TEST_SPLIT, original_folder=CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, new_folder=PROCESSED_CSV_FOLDER_NAME, features_to_drop=FEATURES_TO_DROP)
dp.save_standardized_train_test_datasets()

train_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, train=True, oversample_num=OVERSAMPLE_NUM)
test_ds = PrecipitationClassificationDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, train=False, oversample_num=OVERSAMPLE_NUM)
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
writer = SummaryWriter(log_dir=f"runs/{EXPERIMENT_NAME}")
model = classification_model.Model(NUM_VARIABLES, num_locations, NUM_HIDDENS, BATCH_SIZE)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
# loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([2.0]))
optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
trainer = classification_trainer.Trainer((train_dataloader, test_dataloader), model, loss_fn, optimizer, GRADIENT_CLIP)

best_accuracy = 0
for i in range(EPOCHS):
    epoch = i+1
    print(f"EPOCH: {epoch} \n------------------------------------------")
    train_loss = trainer.train_loop()
    test_loss, test_accuracy = trainer.test_loop()
    baseline_loss, baseline_accuracy = trainer.baseline_loop()
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
    writer.flush()
    print(f"Epoch {epoch} complete! \n--------------------------------")


print("Training complete!")