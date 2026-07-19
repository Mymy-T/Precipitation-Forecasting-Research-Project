import csv
from dataset import PrecipitationClassificationDataset, DatasetType
from enum import Enum
from model import Model
import numpy as np
import os
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from trainer import LSTMTrainer, compute_confusion_matrix, compute_f1
import variables

class ExperimentType(Enum):
    # ONLY 2 possibilites
    main = 0
    hyperparameter_testing = 1
    
class PerformanceLogManager:

    def __init__(self, experiment_type: ExperimentType):
        self.experiment_type = experiment_type
        if experiment_type == ExperimentType.main:
            self.selected_config = variables.MainConfig()
        else:
            self.selected_config = variables.HyperparameterTestingConfig()
        self.data_config = variables.DataConfig()
        self.fixed_hyperparam_config = variables.FixedHyperparameterConfig()
    
    def _prepare_dataloaders(self, lookback, num_locations, dist, draw_num):
        data_file_label = variables.generate_file_label(num_locations=num_locations, dist=dist, draw_num=draw_num) # Latter 2 params don't matter if num_locations=1
        train_ds = PrecipitationClassificationDataset(folder=self.data_config.PROCESSED_CSV_FOLDER_NAME, file_label=data_file_label, lookback=lookback, dataset_type=DatasetType.train)
        valid_ds = PrecipitationClassificationDataset(folder=self.data_config.PROCESSED_CSV_FOLDER_NAME, file_label=data_file_label, lookback=lookback, dataset_type=DatasetType.validation)
        test_ds = PrecipitationClassificationDataset(folder=self.data_config.PROCESSED_CSV_FOLDER_NAME, file_label=data_file_label, lookback=lookback, dataset_type=DatasetType.test)

        running_on_cpu = (variables.DEVICE.type == "cpu")
        train_dataloader = DataLoader(train_ds, batch_size=self.fixed_hyperparam_config.BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=(not running_on_cpu))
        valid_dataloader = DataLoader(valid_ds, batch_size=self.fixed_hyperparam_config.BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(not running_on_cpu))
        test_dataloader = DataLoader(test_ds, batch_size=self.fixed_hyperparam_config.BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(not running_on_cpu))
        return (train_dataloader, valid_dataloader, test_dataloader)

    def _convert_dataloader_to_numpy(self, dataloader):
        """Iterates through a DataLoader to collect and return X and y as NumPy arrays for use in training scikit-learn models."""
        x_batches = []
        y_batches = []
        
        for X, y in dataloader:
            # LSTMs can take an extra lookback dimension in input; logistic model can't, so flattening is needed; X has shape (batch_num, lookback, num_feature_cols)
            batch_size = X.shape[0]
            X = X.reshape(batch_size, -1) # Shape now (batch_num, lookback * num_feature_cols)
            x_batches.append(X.cpu().detach().numpy())
            y_batches.append(y.cpu().detach().numpy())
            
        # Combine all batches
        X_all = np.concatenate(x_batches, axis=0)
        y_all = np.concatenate(y_batches, axis=0)
        
        return X_all, y_all
    

    def _compute_logistic_model_metrics(self, preds, labels):
        fn, fp, tn, tp = compute_confusion_matrix(preds.squeeze(), labels)
        acc = (tp + tn) / labels.shape[0]
        f1 = compute_f1(tp, fn, fp)
        return acc, f1
        
    def train_logistic_model(self, lookback, num_locations, dist, draw_num, run_name):
        train_dataloader, valid_dataloader, test_dataloder = self._prepare_dataloaders(lookback=lookback, num_locations=num_locations, dist=dist, draw_num=draw_num)
        train_X, train_y = self._convert_dataloader_to_numpy(train_dataloader)
        valid_X, valid_y = self._convert_dataloader_to_numpy(valid_dataloader)
        test_X, test_y = self._convert_dataloader_to_numpy(test_dataloder)

        model = LogisticRegression(random_state=1)
        model.fit(train_X, train_y)
        # train_preds = model.predict(train_X)
        valid_preds = model.predict(valid_X)
        test_preds = model.predict(test_X)
        # train_acc, train_f1 = self._compute_logistic_model_metrics(train_preds, train_y)
        # valid_acc, valid_f1 = self._compute_logistic_model_metrics(valid_preds, valid_y)
        test_acc, test_f1 = self._compute_logistic_model_metrics(test_preds, test_y)
        valid_acc, valid_f1 = self._compute_logistic_model_metrics(valid_preds,valid_y)
        print(f"{run_name} completed with test accuracy of {test_acc:.4f} and test f1 of {test_f1:.4f}")
        return test_f1, valid_f1


    def train(self, lookback, num_hiddens, weight_decay, num_locations, dist, draw_num, train_seed, run_name):
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        dataloaders = self._prepare_dataloaders(lookback=lookback, num_locations=num_locations, dist=dist, draw_num=draw_num) # Returns tuple containing (train, valid, test) dataloaders
        writer = SummaryWriter(log_dir=f"{self.selected_config.RUNS_FOLDER}/{run_name}")
        model = Model(self.data_config.NUM_VARIABLES, num_locations, num_hiddens).to(variables.DEVICE)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.fixed_hyperparam_config.WET_DAY_LOSS_WEIGHT]).to(variables.DEVICE))
        optimizer = torch.optim.SGD(model.parameters(), self.fixed_hyperparam_config.LEARNING_RATE, weight_decay=weight_decay)
        trainer = LSTMTrainer(dataloaders, model, loss_fn, optimizer, self.fixed_hyperparam_config.GRADIENT_CLIP, printing_metrics=False)

        model_folder_path = os.path.join(os.getcwd(), self.selected_config.MODEL_SAVE_FOLDER)
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        model_save_path = os.path.join(model_folder_path, f"model__{run_name}")

        best_valid_fscore = -float("inf") # For early stopping using the validation set
        counter = 0 # Counts number of times valid_fscore didn't improve from last epoch

        for epoch in range(1, self.fixed_hyperparam_config.MAX_EPOCHS + 1):
            train_loss, train_accuracy, train_f1 = trainer.train_loop()
            valid_loss, valid_accuracy, valid_f1 = trainer.test_loop(valid=True)
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Train": train_loss, "Valid": valid_loss}, global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"Train": train_accuracy, "Valid": valid_accuracy}, global_step=epoch)
            writer.add_scalars(main_tag="F1 Score", tag_scalar_dict={"Train": train_f1, "Valid": valid_f1}, global_step=epoch)
            writer.flush()

            if valid_f1 > best_valid_fscore:
                best_valid_fscore = valid_f1
                counter = 0
                torch.save(model.state_dict(), model_save_path) # Saves the best version; all other later saves override the previous model save
            else:
                if epoch <= self.fixed_hyperparam_config.MIN_EPOCHS:
                    continue
                counter += 1

            if counter >= self.fixed_hyperparam_config.EARLY_STOPPING_PATIENCE:
                print(f"[{run_name}] stopped early, on epoch {epoch} of {self.fixed_hyperparam_config.MAX_EPOCHS}")
                break

        model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=variables.DEVICE)) # Load the best model
        _, test_accuracy, test_f1 = trainer.test_loop(valid=False)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"Test": test_accuracy}, global_step=1)
        writer.add_scalars(main_tag="F1 Score", tag_scalar_dict={"Test": test_f1}, global_step=1)
        writer.flush()
        writer.close()
        return test_f1, best_valid_fscore # best_valid_fscore is for best checkpoint; used for hyperparam testing

    def _load_completed_keys(self, baseline=False):
        path = os.path.join(os.getcwd(), self.selected_config.RESULTS_PATH if not baseline else variables.MainConfig().BASELINE_RESULTS_PATH)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {}
        existing = pd.read_csv(path)
        results_cols = self.selected_config.RESULTS_COLUMNS if not baseline else variables.MainConfig().RESULTS_COLUMNS
        if results_cols == variables.MainConfig().RESULTS_COLUMNS:
            key_cols = results_cols[:4] # num_locations, dist, draw_num, seed
            value_cols = results_cols[4:] # F1
        else:
            key_cols = results_cols[:5] # num_locations, dist, lookback, num_hiddens, weight_decay
            value_cols = results_cols[5:] # valid_F1, F1
        keys = zip(*(existing[col] for col in key_cols)) # [(first_key_0, second_key_0, third_key_0, ...), (first_key_1, second_key_1, third_key_1, ...), ...]
        if len(value_cols) == 1:
            values = existing[value_cols[0]] # [f1_0, f1_1, ...]
        else:
            values = zip(*(existing[col] for col in value_cols)) # [(valid_f1_0, f1_0), (valid_f1_1, f1_1), ...]
        return dict(zip(keys, values)) # Of the form {key_tuple: value_or_tuple_of_values, ...}

    def _append_result(self, row, baseline=False):
        path = os.path.join(os.getcwd(), self.selected_config.RESULTS_PATH if not baseline else variables.MainConfig().BASELINE_RESULTS_PATH)
        file_exists = os.path.exists(path)
        fieldnames = self.selected_config.RESULTS_COLUMNS if not baseline else variables.MainConfig().RESULTS_COLUMNS
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader() # Header columns
            writer.writerow(row) # Write a new row of data

    def run_and_log(self, lookback, num_hiddens, weight_decay, num_locations, dist, draw_num, train_seed, run_name, baseline=False):
        # key must match the exact order and num of cols as self.selected_config.Results
        if self.experiment_type == ExperimentType.main:
            key = (num_locations, dist, draw_num, train_seed)
        else:
            key = (num_locations, dist, lookback, num_hiddens, weight_decay)
        completed = self._load_completed_keys(baseline) # This returns a list of all the information for prior runs; if another run happens to match the label data, it has already been done and should not be run again
        results_cols = self.selected_config.RESULTS_COLUMNS if not baseline else variables.MainConfig().RESULTS_COLUMNS
        if key in completed:
            prior_data = completed[key]
            if self.experiment_type == ExperimentType.hyperparameter_testing:
                prior_valid_f1, prior_test_f1 = prior_data
                if pd.isna(prior_valid_f1) or pd.isna(prior_test_f1):
                    raise ValueError(f"{key} exists in history but has empty valid_F1 or F1 field! See {os.path.join(os.getcwd(), self.selected_config.RESULTS_PATH)}")
                return prior_test_f1, prior_valid_f1
            else:
                if pd.isna(prior_data):
                    raise ValueError(f"{key} exists in history but has empty F1 field! See {os.path.join(os.getcwd(), self.selected_config.RESULTS_PATH)}")
                return prior_data, None
        if not baseline:
            test_f1, valid_f1 = self.train(lookback=lookback, num_hiddens=num_hiddens, weight_decay=weight_decay, num_locations=num_locations, dist=dist, draw_num=draw_num, train_seed=train_seed, run_name=run_name)
        else:
            test_f1, valid_f1 = self.train_logistic_model(lookback=lookback, num_locations=num_locations, dist=dist, draw_num=draw_num, run_name=run_name)
        if self.experiment_type == ExperimentType.main:
            row = {results_cols[0]: num_locations, results_cols[1]: dist, results_cols[2]: draw_num, results_cols[3]: train_seed, results_cols[4]: test_f1}
        else:
            row = {results_cols[0]: num_locations, results_cols[1]: dist, results_cols[2]: lookback, results_cols[3]: num_hiddens, results_cols[4]: weight_decay, results_cols[5]: valid_f1, results_cols[6]: test_f1}
        print(f"{run_name} F1: {test_f1:.4f}")
        self._append_result(row, baseline)
        return test_f1, valid_f1
