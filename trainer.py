import torch

class Trainer:
    def __init__(self, dataloaders, model, loss_fn, optimizer, grad_clip):
        self.train_dataloader, self.test_dataloader = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def _compute_confusion_matrix(self, prob_pred, target):
        false_negatives = 0
        false_positives = 0
        true_negatives = 0
        true_positives = 0
        for i in range(prob_pred.shape[0]):
            if prob_pred[i] >= 0.5:
                pred_class = 1
            else:
                pred_class = 0
            
            if (pred_class == 0) and (target[i] == 1):
                false_negatives += 1
            elif (pred_class == 1) and (target[i] == 0):
                false_positives += 1
            elif (pred_class == 0) and (target[i] == 0):
                true_negatives += 1
            elif (pred_class == 1) and (target[i] == 1):
                true_positives += 1
        
        return false_negatives, false_positives, true_negatives, true_positives



    
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
            fn_tot = 0
            fp_tot = 0
            tn_tot = 0
            tp_tot = 0
            for batch_num, (X, y) in enumerate(self.test_dataloader):
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
                prob_pred = torch.sigmoid(pred)
                # for i in range(prob_pred.shape[0]):
                fn, fp, tn, tp = self._compute_confusion_matrix(prob_pred, y)
                fn_tot += fn
                fp_tot += fp
                tn_tot += tn
                tp_tot += tp
                #     if prob_pred[i] >= 0.5:
                #         pred_class = 1
                #     else:
                #         pred_class = 0
                #     if pred_class == y[i]:
                #         total_correct += 1
            
            avg_loss = total_loss / len(self.test_dataloader) # tot_loss divided by num batches
            accuracy = (tn_tot + tp_tot) / (len(self.test_dataloader) * prob_pred.shape[0])
            recall = tp_tot / (tp_tot + fn_tot)
            precision = tp_tot / (tp_tot + fp_tot)
            f1 = 2*tp_tot / (2*tp_tot + fn_tot + fp_tot)
            print(f"Avg loss per batch test set: {avg_loss}")
            print(f"Accuracy: {accuracy * 100}%")
            print(f"Recall: {recall * 100}%")
            print(f"Precision: {precision * 100}%")
            print(f"f1: {f1}")
            return avg_loss, accuracy, f1
    
    def baseline_loop(self):
        total_loss = 0
        fn_tot = 0
        fp_tot = 0
        tn_tot = 0
        tp_tot = 0
        for batch_num, (X, y) in enumerate(self.test_dataloader):
            pred = torch.ones_like(y) * -10 # Simulate predicting 0
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            prob_pred = torch.sigmoid(pred)
            fn, fp, tn, tp = self._compute_confusion_matrix(prob_pred, y)
            fn_tot += fn
            fp_tot += fp
            tn_tot += tn
            tp_tot += tp
            # for i in range(prob_pred.shape[0]):
            #     if prob_pred[i] >= 0.5:
            #         pred_class = 1
            #     else:
            #         pred_class = 0
            #     if pred_class == y[i]:
            #         total_correct += 1
        avg_loss = total_loss / len(self.test_dataloader) # tot_loss divided by num batches
        accuracy = (tn_tot + tp_tot) / (len(self.test_dataloader) * prob_pred.shape[0])

        recall = tp_tot / (tp_tot + fn_tot)
        f1 = 2*tp_tot / (2*tp_tot + fn_tot + fp_tot)
        print(f"Basline avg loss per batch test set: {avg_loss}")
        print(f"Baseline Accuracy: {accuracy * 100}%")
        print(f"Baseline Recall: {recall * 100}%")
        print(f"Baseline Precision: undefined")
        print(f"Baseline f1: {f1}")
        return avg_loss, accuracy, f1

