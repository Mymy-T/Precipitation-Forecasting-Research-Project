import torch

def compute_f1(tp, fn, fp):
    denom = 2 * tp + fn + fp
    if denom == 0:
        return 0.0
    return 2 * tp / denom

def compute_confusion_matrix(prob_pred, target):
    # Only allows for one timestep ahead forecasting right now
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

class LSTMTrainer:
    def __init__(self, dataloaders, model, loss_fn, optimizer, grad_clip, printing_metrics):
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.printing = printing_metrics


    def train_loop(self):
        self.model.train()
        total_loss = 0
        total_examples = 0
        fn_tot = 0
        fp_tot = 0
        tn_tot = 0
        tp_tot = 0
        for batch_num, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            # Report loss before stepping; get values to determine f score
            total_loss += loss.item()
            prob_pred = torch.sigmoid(pred)
            total_examples += prob_pred.shape[0]
            fn, fp, tn, tp = compute_confusion_matrix(prob_pred, y)
            fn_tot += fn
            fp_tot += fp
            tn_tot += tn
            tp_tot += tp
            if batch_num % 30 == 0:
                    print(loss.item())
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = (tn_tot + tp_tot) / total_examples
        f1 = compute_f1(tp_tot, fn_tot, fp_tot)
        if self.printing:
            print(f"Train Loss: {avg_loss}")
            print(f"Train Accuracy: {accuracy * 100}%")
            print(f"Train f1: {f1}")
        return avg_loss, accuracy, f1


    def test_loop(self, valid):
        self.model.eval()
        if valid:
            dl = self.valid_dataloader
            name = "Valid"
        else:
            dl = self.test_dataloader
            name = "Test"
        with torch.no_grad():
            total_loss = 0
            total_examples = 0
            fn_tot = 0
            fp_tot = 0
            tn_tot = 0
            tp_tot = 0
            for batch_num, (X, y) in enumerate(dl):
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
                prob_pred = torch.sigmoid(pred)
                total_examples += prob_pred.shape[0]
                fn, fp, tn, tp = compute_confusion_matrix(prob_pred, y)
                fn_tot += fn
                fp_tot += fp
                tn_tot += tn
                tp_tot += tp
            avg_loss = total_loss / len(dl) # tot_loss divided by num batches
            accuracy = (tn_tot + tp_tot) / total_examples
            f1 = compute_f1(tp_tot, fn_tot, fp_tot)
            if self.printing:
                print(f"{name} Loss: {avg_loss}")
                print(f"{name} Accuracy: {accuracy * 100}%")
                print(f"{name} f1: {f1}")
            return avg_loss, accuracy, f1


    def baseline_loop(self):
        total_loss = 0
        total_examples = 0
        fn_tot = 0
        fp_tot = 0
        tn_tot = 0
        tp_tot = 0
        for batch_num, (X, y) in enumerate(self.test_dataloader):
            pred = torch.ones_like(y) * -10 # Simulate predicting 0 for all samples
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            prob_pred = torch.sigmoid(pred)
            total_examples += prob_pred.shape[0]
            fn, fp, tn, tp = compute_confusion_matrix(prob_pred, y)
            fn_tot += fn
            fp_tot += fp
            tn_tot += tn
            tp_tot += tp
        avg_loss = total_loss / len(self.test_dataloader) # tot_loss divided by num batches
        accuracy = (tn_tot + tp_tot) / total_examples
        f1 = compute_f1(tp_tot, fn_tot, fp_tot)
        if self.printing:
            print(f"Basline avg loss per batch test set: {avg_loss}")
            print(f"Baseline Accuracy: {accuracy * 100}%")
            print(f"Baseline f1: {f1}")
        return avg_loss, accuracy, f1

    def random_guessing_loop(self):
        total_examples = 0
        fn_tot = 0
        fp_tot = 0
        tn_tot = 0
        tp_tot = 0
        for batch_num, (X, y) in enumerate(self.test_dataloader):
            prob_pred = torch.rand_like(y) # Simulate random prediction between 0 and 1
            total_examples += prob_pred.shape[0]
            fn, fp, tn, tp = compute_confusion_matrix(prob_pred, y)
            fn_tot += fn
            fp_tot += fp
            tn_tot += tn
            tp_tot += tp

        accuracy = (tn_tot + tp_tot) / total_examples
        f1 = compute_f1(tp_tot, fn_tot, fp_tot)
        if self.printing:
            print(f"Random Accuracy: {accuracy * 100}%")
            print(f"Random f1: {f1}")
        return accuracy, f1
