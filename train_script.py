import dataset
import model
import trainer
import torch
from torch.utils.data import DataLoader, random_split

NUM_VARIABLES = 5
FORECAST_STEPS = 6
BATCH_SIZE = 8
NUM_HIDDENS = 16
LEARNING_RATE = 1e-3
EPOCHS = 2

num_locations = 16

wds = dataset.WeatherDataset("CSV_Datasets", "16-sites__100-miles.csv", 24, 6)
training_ds, test_ds = random_split(wds, [0.8, 0.2])
train_dataloader = DataLoader(training_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model = model.Model(NUM_VARIABLES, num_locations, NUM_HIDDENS, FORECAST_STEPS, BATCH_SIZE)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
trainer = trainer.Trainer((train_dataloader, test_dataloader), model, loss_fn, optimizer)


for i in range(EPOCHS):
    print(f"EPOCH: {i+1} \n------------------------------------------")
    trainer.train_loop()
    trainer.test_loop()
    print(f"Epoch {i+1} complete! \n--------------------------------")

print("Training complete!")
