import dataset
import preprocessor
import model
import trainer
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

CSV_FOLDER_NAME = "CompleteDatasets"
PROCESSED_CSV_FOLDER_NAME = "Split_Datasets"
EXPERIMENT_NAME = "classify"

NUM_VARIABLES = 11
LOOKBACK = 4
FORECAST_STEPS = 1
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 1
NUM_HIDDENS = 256
LEARNING_RATE = 1e-3
EPOCHS = 25
GRADIENT_CLIP = 1

num_locations = 10
distance = 100

dp = preprocessor.DataPreprocessor(train_proportion=TRAIN_TEST_SPLIT, original_folder=CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, new_folder=PROCESSED_CSV_FOLDER_NAME, features_to_drop=[])
dp.save_standardized_train_test_datasets()

train_wds = dataset.WeatherDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, num_forecast_steps=FORECAST_STEPS, train=True, max_zeros=5)
test_wds = dataset.WeatherDataset(folder=PROCESSED_CSV_FOLDER_NAME, num_sites=num_locations, dist=distance, lookback=LOOKBACK, num_forecast_steps=FORECAST_STEPS, train=False, max_zeros=5)
train_dataloader = DataLoader(train_wds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_wds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
writer = SummaryWriter(log_dir=f"runs/{EXPERIMENT_NAME}")
model = model.Model(NUM_VARIABLES, num_locations, NUM_HIDDENS, FORECAST_STEPS, BATCH_SIZE)
# loss_fn = torch.nn.MSELoss() # MSE switch back to later
loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([2.0]))
optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
trainer = trainer.Trainer((train_dataloader, test_dataloader), model, loss_fn, optimizer, GRADIENT_CLIP, writer=writer)

for i in range(EPOCHS):
    print(f"EPOCH: {i+1} \n------------------------------------------")
    train_loss = trainer.train_loop(epoch_num=i+1)
    test_loss = trainer.test_loop(epoch_num=i+1)
    baseline_loss = trainer.baseline(epoch_num=i+1)
    writer.add_scalars(main_tag="Loss",
        tag_scalar_dict={
            "Train":train_loss,
            "Test":test_loss,
            "Baseline":baseline_loss
        },
        global_step=i+1
    )
    writer.flush()
    print(f"Epoch {i+1} complete! \n--------------------------------")


print("Training complete!")