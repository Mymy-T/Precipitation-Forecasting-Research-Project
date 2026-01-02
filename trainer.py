import torch

class Trainer:
    def __init__(self, dataloaders, model, loss_fn, optimizer):
        self.train_dataloader, self.test_dataloader = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def train_loop(self):
        self.model.train()
        for batch_num, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch_num % 100 == 0:
                print(f"Batch number {batch_num} with avg loss {loss.item()}")
            if batch_num == len(self.train_dataloader):
                print(f"Training loop complete")
    
    def test_loop(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for X, y in self.test_dataloader:
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.test_dataloader) # tot_loss divided by num batches
            print(f"Avg loss on test set: {avg_loss}")
