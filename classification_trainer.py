import torch

class Trainer:
    def __init__(self, dataloaders, model, loss_fn, optimizer, grad_clip):
        self.train_dataloader, self.test_dataloader = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_clip = grad_clip
    
    def train_loop(self):
        self.model.train()
        total_loss = 0
        for batch_num, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            # Report loss before stepping
            total_loss += loss.item()
            if batch_num % 10 == 0:
                    print(loss.item())
        return total_loss/len(self.train_dataloader) # avg loss at the end of the epoch over all batches
               
    
    def test_loop(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for batch_num, (X, y) in enumerate(self.test_dataloader):
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
                prob_pred = torch.sigmoid(pred)
                for i in range(prob_pred.shape[0]):
                    if prob_pred[i] >= 0.5:
                        pred_class = 1
                    else:
                        pred_class = 0
                    if pred_class == y[i]:
                        total_correct += 1
            
            avg_loss = total_loss / len(self.test_dataloader) # tot_loss divided by num batches
            accuracy = total_correct / (len(self.test_dataloader) * prob_pred.shape[0])
            print(f"Avg loss per batch test set: {avg_loss}")
            print(f"Accuracy: {accuracy * 100}%")
            return avg_loss, accuracy
    
    def baseline_loop(self):
        total_loss = 0
        total_correct = 0
        for batch_num, (X, y) in enumerate(self.test_dataloader):
            pred = torch.ones_like(y) * -100 # Simulate predicting 0
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            prob_pred = torch.sigmoid(pred)
            for i in range(prob_pred.shape[0]):
                if prob_pred[i] >= 0.5:
                    pred_class = 1
                else:
                    pred_class = 0
                if pred_class == y[i]:
                    total_correct += 1
        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = total_correct / (len(self.test_dataloader) * prob_pred.shape[0])
        print(f"Baseline Avg loss per batch on test set: {avg_loss}")
        print(f"Baseline Accuracy: {accuracy * 100}%")
        return (avg_loss, accuracy)

