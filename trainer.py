import matplotlib
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, dataloaders, model, loss_fn, optimizer, grad_clip, writer: SummaryWriter):
        self.train_dataloader, self.test_dataloader = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        # self.writer = writer
    
    def train_loop(self, epoch_num):
        self.model.train()
        for batch_num, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            print(loss.item())
            self.optimizer.step()
            # self.writer.add_scalar("Train_Loss", loss, (epoch_num - 1)*len(self.train_dataloader) + batch_num)
            if batch_num % 100 == 0:
                print(f"Batch number {batch_num} with avg loss {loss.item()}")
                # print(f"Grad Norm (before clip): {grad_norm}")
            if batch_num == len(self.train_dataloader)-1:
                print(f"Training loop complete")
                return loss
        # self.writer.flush()
               
    
    def test_loop(self, epoch_num):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch_num, (X, y) in enumerate(self.test_dataloader):
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
                # self.writer.add_scalar("Test_Loss", loss, (epoch_num - 1)*len(self.test_dataloader) + batch_num)
            
            avg_loss = total_loss / len(self.test_dataloader) # tot_loss divided by num batches
            print(f"Avg loss per batch test set: {avg_loss}")
            # self.writer.flush()
            return loss # Loss on final batch of epoch
    
    def baseline(self, epoch_num):
        total_loss = 0
        for batch_num, (X, y) in enumerate(self.test_dataloader):
            pred = torch.zeros_like(y)
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            # self.writer.add_scalar("Baseline_Test_Loss", loss, (epoch_num - 1)*len(self.test_dataloader) + batch_num)
        # self.writer.flush()
        avg_loss = total_loss / len(self.test_dataloader) # tot_loss divided by num batches
        print(f"Avg baseline loss per batch test set: {avg_loss}")

        return loss # Loss on the final batch of epoch

