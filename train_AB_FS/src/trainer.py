from typing import Any, Union, Dict, List

import sys
import os
import csv
import pandas as pd
import seaborn as sns
from time import time
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    auc,
    roc_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from src.modular_rimnet_mod import ModularRimNet #********************************
from src.loggers import AbstractLogger, FileLogger
from src.utils import empty_cache, collate_outputs, flatten_nested_numbers
from src.SMSC import BATCHKEYS


class BasicRimNetTrainer:
    def __init__(
        self,
        model: ModularRimNet,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        task: str,
        output_path: str,
        loss_fn=None,
        optimizer=None,
        lr_scheduler=None,
        device=None,
        num_epochs: int = 50,
        num_iterations_per_epoch: int = None,
        num_val_iterations_per_epoch: int = None,
        save_every_epochs=10,
        #device: torch.device = torch.device("cuda"),
        logger: AbstractLogger = None,
        ema_measure="F1",
    ):
        # dataloader for training and validation
        assert isinstance(train_dataloader, DataLoader)
        assert isinstance(val_dataloader, DataLoader)
        assert isinstance(
            model, ModularRimNet
        ), f"The model: {model} is not a ModularRimNet"  # this is important for the input spliting
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_val_iterations_per_epoch = (
            len(val_dataloader.dataset) // val_dataloader.batch_size
            if num_val_iterations_per_epoch is None
            else num_val_iterations_per_epoch
        )
        self.num_epochs = num_epochs
        self.num_iterations_per_epoch = (
            len(train_dataloader.dataset) // train_dataloader.batch_size
            if num_iterations_per_epoch is None
            else num_iterations_per_epoch
        )
        self.current_epoch = 0
        self.save_every = save_every_epochs
        self.ema_measure = ema_measure
        self.best_ema = None
        self.epoch_ema = None
        self.best_loss = None
        self.task = task
        self.output_folder = output_path
        self.logger = logger
        self.csv_summary = None

        self.loss = loss_fn
        self.optimizer = (
            optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1)
            if optimizer is None
            else optimizer
        )
        self.lr_scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.1, patience=10
            )
            if lr_scheduler is None
            else lr_scheduler
        )

        self.step_loss = None
        self.step_val_loss = None
        self.mean_epoch_loss = None
        self.mean_val_epoch_loss = None

    def train(self):
        epoch_tqdm_log = tqdm(total=0, position=0, bar_format="{desc}")
        self.train_start()
        #print(self.output_folder)
        #print(self.logger)
        for epoch in trange(self.current_epoch, self.num_epochs):
            # epoch starts
            self.model.train()
            #print('check 1') 
            train_outputs = []
            for batch_idx, batch in enumerate(self.train_dataloader):
                #print(batch_idx)
                #print(self.num_iterations_per_epoch)
                
                if batch_idx >= self.num_iterations_per_epoch:
                    continue
                train_outputs.append(self.train_step(batch))

            # average loss at the current epoch
            self.mean_epoch_loss = np.mean(collate_outputs(train_outputs)["loss"])
            self.train_epoch_end_log()
            #print('check 2')

            # Define the path to the CSV file
            csv_file = f"./test_details.csv"
            df = pd.DataFrame(columns=["subject", "lesion", "type"])
            df.to_csv(csv_file, index=False)

            # Validation block
            with torch.no_grad():
                self.model.eval()
                val_outputs = []
                for batch_idx, batch in enumerate(self.val_dataloader):
                    if batch_idx >= self.num_val_iterations_per_epoch:
                        continue
                    val_outputs.append(self.validation_step(csv_file, batch))

                if self.task == "multiclass":
                    self.validation_epoch_end_multi(val_outputs)
                else:
                    self.validation_epoch_end(val_outputs)
                self.mean_val_epoch_loss = np.mean(collate_outputs(val_outputs)["loss"])
            self.scheduler_step()  # Depending on the scheduler could need not available infor #TODO
            #print('check 3')

            epoch_tqdm_log.set_description_str(
                f"Epoch: {self.current_epoch} | Train loss {self.mean_epoch_loss} | Val. loss {self.mean_val_epoch_loss} | Current {self.ema_measure} {self.epoch_ema} | Best {self.ema_measure} {self.best_ema}"
            )
            #print('check 4')
            self.epoch_end()

        self.train_end()

    def train_start(self):
        timestamp = datetime.now()
        folder_name = f"{timestamp.day}_{timestamp.month}_{timestamp.year}_{timestamp.hour}_{timestamp.minute}_{timestamp.second}"
        self.output_folder = os.path.join(
            self.output_folder,
            f"exp_{self.model.__class__.__name__}_{folder_name}",
        )
        os.makedirs(self.output_folder, exist_ok=True)
        
    
        if self.logger is None:
            self.logger = FileLogger(
                os.path.join(self.output_folder, f"log_{folder_name}.log")
            )
        self.logger.log("Start Training")
        self.logger.log(str(self.model))

        self.model = self.model.to(self.device)
        empty_cache(self.device)

    def epoch_start_log(self):
        self.logger.log("epoch_start_timestamps", time(), self.current_epoch)
        self.logger.log(f"Epoch {self.current_epoch}")
        self.logger.log(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )

    def train_step(self, batch: dict) -> dict:
        data = batch[BATCHKEYS.IMAGE].to(
            self.device, non_blocking=True
        )  # TODO the imposition of a dataset with this labels and structure
        #print(list(data.shape[1:]))
        expected_dimensions = [self.model.n_modalities] + [
            s for s in self.model.input_size
        ]
        
        assert (
            list(data.shape[1:]) == expected_dimensions
        ), f"The expected model dimensions {expected_dimensions} are different to the input dimensions list(data.shape[1:])"
        target = batch[BATCHKEYS.LABEL].to(self.device, non_blocking=True)
        
        #print ("batch index ... 0/1: {}/{}".format(
        #        len(np.where(target.cpu().numpy() == 0)[0]),
        #        len(np.where(target.cpu().numpy() == 1)[0])))
        
        self.optimizer.zero_grad(set_to_none=True)

        output = self.model(data)
        l = self.loss(output, target)
        self.step_loss = l.clone()
        l.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
        self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}  # usually batch averaged

    def train_epoch_end_log(self):
        self.logger.log(
            f"Epoch:{self.current_epoch} Train loss: {self.mean_epoch_loss}"
        )

    def validation_step(self, csv_file, batch: dict) -> dict:
        data = batch[BATCHKEYS.IMAGE]  # TODO the imposition of a dataset with this labels and structure
        target = batch[BATCHKEYS.LABEL]

        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        output = self.model(data)
        del data
        l = self.loss(output, target)
        self.step_val_loss = l.clone()
        
        #########
        print_val = False
        ######### 
        
        # Iterate through the batch to determine the result type for each sample
        if print_val == True:
          predict_proba = torch.softmax(output, dim=1)
          output_binary = torch.argmax(predict_proba, dim=1)
        
          # Convert output and target to binary decisions (0 or 1)
          output_binary = output_binary.cpu().tolist()
          target_binary = target.detach().cpu().tolist()

          # Prepare a list to store results
          results = []
          
          for i in range(len(output_binary)):
            subject = str(batch[BATCHKEYS.SUBJECT][i])  # Extract subject
            lesion = str(batch[BATCHKEYS.LESION].tolist()[i])  # Extract lesion

            # Determine the result type based on the conditions
            if output_binary[i] == 1 and target_binary[i] == 1:
                result_type = "TP"  # True Positive
            elif output_binary[i] == 0 and target_binary[i] == 1:
                result_type = "FN"  # False Negative
            elif output_binary[i] == 1 and target_binary[i] == 0:
                result_type = "FP"  # False Positive
            else:
                result_type = "TN"  # True Negative

            # Append the result for the current sample
            results.append([subject, lesion, result_type])

          # Convert the results to a DataFrame and append to the CSV file
          df = pd.DataFrame(results, columns=["subject", "lesion", "type"])
          df.to_csv(csv_file, mode='a', header=False, index=False)

        if self.task == "multiclass":
            return {
            **{"loss": l.detach().cpu().numpy()},
            **self.compute_validation_metrics_multi(target, output),
        }
        else:
            return {
            **{"loss": l.detach().cpu().numpy()},
            **self.compute_validation_metrics(target, output),
        }

    # This is the definition for LRPlateau. Extend the class for a different Scheduler
    def scheduler_step(self):
        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(self.mean_epoch_loss)
        else:
            self.lr_scheduler.step()

    def compute_validation_metrics(self, target, model_output) -> Dict:
        
        model_output = model_output.view(-1, 2)  # Reshaping to (N, 2), N is the number of samples

        # Apply softmax to get probabilities for both classes
        predict_proba = torch.softmax(model_output, dim=1)
        
        # Convert to binary output (take the index of the class with the highest probability)
        binary_outputs = torch.argmax(predict_proba, dim=1)
        
        predict_proba = predict_proba[:, 1]
        
        #predict_proba, binary_outputs = torch.max(torch.softmax(model_output, dim=1), dim=1)
        confusion_matrix = ConfusionMatrix(
            task="binary", num_classes=model_output.shape[1]
        ).to(self.device)(binary_outputs, target)
        tn = confusion_matrix[0][0].detach().cpu().numpy()
        fp = confusion_matrix[0][1].detach().cpu().numpy()
        fn = confusion_matrix[1][0].detach().cpu().numpy()
        tp = confusion_matrix[1][1].detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        binary_outputs = binary_outputs.detach().cpu().numpy()
        predict_proba = predict_proba.detach().cpu().numpy()
        precision, recall, _ = precision_recall_curve(target, binary_outputs)
        monoclass_batch = len(np.unique(target)) == 1
        #auc_val = (
        #    roc_auc_score(target, predict_proba) if not monoclass_batch else np.nan
        #)  # not valid with just one class
        # pr_auc = auc(target, predict_proba)  if not monoclass_batch  else np.nan # not valid with just one class
        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "specificity": tn / (tn + fp),
            "fpr": fp / (fp + tn),
            "precision_val": precision_score(target, binary_outputs),
            "recall": recall_score(target, binary_outputs, average="micro"),
            "F1": f1_score(target, binary_outputs),
            "accuracy": accuracy_score(target, binary_outputs),
            "predictions": predict_proba,
            "targets": target,
        }

    def validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp"])
        fp = np.sum(outputs_collated["fp"])
        fn = np.sum(outputs_collated["fn"])
        tn = np.sum(outputs_collated["tn"])

        val_loss = np.mean(outputs_collated["loss"])
        self.mean_val_epoch_loss = val_loss

        global_f1_per_class = [
            i for i in [2 * i / (2 * i + j + k) for i, j, k in zip([tp], [fp], [fn])]
        ]
        mean_f1_dice = np.nanmean(global_f1_per_class)
        self.epoch_ema = mean_f1_dice
        if self.best_ema is None or self.epoch_ema > self.best_ema:
            self.best_ema = self.epoch_ema
            print(f"New best EMA {self.ema_measure} : {np.round(self.best_ema, decimals=4)}")
            self.save_checkpoint(
                #os.path.join(self.output_folder, f"checkpoint_best_Epoch{self.current_epoch}.pth")
                os.path.join(self.output_folder, f"checkpoint_best_Epoch.pth")
            )
        if self.best_loss is None or self.mean_val_epoch_loss < self.best_loss:
            self.best_loss = self.mean_val_epoch_loss
            print(f"New best EMA Loss : {np.round(self.best_loss, decimals=4)}")
            self.save_checkpoint(
                #os.path.join(self.output_folder, f"checkpoint_best_Epoch{self.current_epoch}.pth")
                os.path.join(self.output_folder, f"checkpoint_best_Loss_Epoch.pth")
            )

        # Accumulate all predictions and targets for AUC calculation
        all_predictions = np.concatenate([pred.flatten() for pred in outputs_collated["predictions"]])
        all_targets = np.concatenate([pred.flatten() for pred in outputs_collated["targets"]])
        
        # Calculate overall AUC on the entire validation set
        overall_auc = roc_auc_score(all_targets, all_predictions)
        
        fpr, tpr, thresholds = roc_curve(all_targets, all_predictions)

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {overall_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random guessing
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"{self.output_folder}/ROC_best_Epoch")
        
        # Since the scheduler could be None, its safer to acces lr this way
        lr = self.optimizer.param_groups[0]["lr"]
        if self.csv_summary is None:
            self.create_summary(
                ["epoch", "train_loss", 'val_loss', self.ema_measure, 'AUC', 'tp_t', 'fp_t', 'fn_t', 'tn_t', 'lr']
            )
            
        self.append_metrics_to_summary(
            [self.current_epoch, self.mean_epoch_loss, self.mean_val_epoch_loss, self.epoch_ema, overall_auc, tp, fp, fn, tn, lr]
        )

    def epoch_end(self):
        self.logger.log(
            f"Epoch {self.current_epoch} Val. loss {self.mean_val_epoch_loss}"
        )

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(
                os.path.join(self.output_folder, "checkpoint_latest.pth")
            )
        self.current_epoch += 1

    def train_end(self):
        self.logger.log("End training")
        empty_cache(self.device)
        # self.print_to_log_file("Training done.")

    def save_checkpoint(self, filename: str) -> None:
        mod = self.model
        checkpoint = {
            "network_weights": mod.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_ema": self.best_ema,
            "current_epoch": self.current_epoch + 1,
            "trainer_name": self.__class__.__name__,
            "model_name": self.model.__class__.__name__,
        }
        torch.save(checkpoint, filename)

    def create_summary(self, columns):
        self.csv_summary = os.path.join(self.output_folder, "summary.csv")
        with open(self.csv_summary, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(columns)

    def append_metrics_to_summary(self, metrics):
        """
        Append a line with metrics to an existing CSV file.

        Parameters:
        - file_path (str): Path to the CSV file.
        - metrics (list): List of metric values.
        """
        with open(self.csv_summary, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(metrics)
            
    def compute_validation_metrics_multi(self, target, model_output) -> Dict:
        num_classes = 3  # Adjusted for multi-class classification
        
        model_output = model_output.view(-1, num_classes)  # Reshaping to (N, num_classes)
        predict_proba = torch.softmax(model_output, dim=1)  # Compute probabilities
        binary_outputs = torch.argmax(predict_proba, dim=1)  # Get class predictions
        
        conf_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        ).to(self.device)(binary_outputs, target)
        
        conf_matrix = conf_matrix.detach().cpu().numpy()

        target = target.detach().cpu().numpy()
        binary_outputs = binary_outputs.detach().cpu().numpy()
        predict_proba = predict_proba.detach().cpu().numpy()
        
        # Extract TP, TN, FP, FN per class
        tp = np.diag(conf_matrix)
        fn = conf_matrix.sum(axis=1) - tp
        fp = conf_matrix.sum(axis=0) - tp
        tn = conf_matrix.sum() - (tp + fp + fn)
        
        precision = precision_score(target, binary_outputs, average="macro")
        recall = recall_score(target, binary_outputs, average="macro")
        f1 = f1_score(target, binary_outputs, average="macro")
        accuracy = accuracy_score(target, binary_outputs)
        
        return {
            "conf_matrix_row0": conf_matrix[0,:],
            "conf_matrix_row1": conf_matrix[1,:],
            "conf_matrix_row2": conf_matrix[2,:],
            "tp": tp.tolist(),
            "fp": fp.tolist(),
            "fn": fn.tolist(),
            "tn": tn.tolist(),
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "accuracy": accuracy,
            "predictions": predict_proba,
            "targets": target,
        }

    def validation_epoch_end_multi(self, val_outputs: List[dict]):
        
        outputs_collated = collate_outputs(val_outputs)
        
        conf_matrix_row0 = np.array(outputs_collated["conf_matrix_row0"]).reshape(-1, 3).sum(axis=0)
        conf_matrix_row1 = np.array(outputs_collated["conf_matrix_row1"]).reshape(-1, 3).sum(axis=0)
        conf_matrix_row2 = np.array(outputs_collated["conf_matrix_row2"]).reshape(-1, 3).sum(axis=0)
        
        tp = np.array(outputs_collated["tp"]).reshape(-1, 3).sum(axis=0)
        fp = np.array(outputs_collated["fp"]).reshape(-1, 3).sum(axis=0)
        fn = np.array(outputs_collated["fn"]).reshape(-1, 3).sum(axis=0)
        tn = np.array(outputs_collated["tn"]).reshape(-1, 3).sum(axis=0)
             
        val_loss = np.mean(outputs_collated["loss"])
        self.mean_val_epoch_loss = val_loss
        
        all_predictions = outputs_collated["predictions"]
        all_targets = outputs_collated["targets"]
        
        f1_per_class = [2 * tp[i] / (2 * tp[i] + fp[i] + fn[i]) if (2 * tp[i] + fp[i] + fn[i]) > 0 else 0 for i in range(len(tp))]
        mean_f1_dice = np.nanmean(f1_per_class)
        
        predicted_classes = np.argmax(all_predictions, axis=1)
        precision_macro = precision_score(all_targets, predicted_classes, average='macro')
        recall_macro = recall_score(all_targets, predicted_classes, average='macro')
        precision_weighted = precision_score(all_targets, predicted_classes, average='weighted')
        recall_weighted = recall_score(all_targets, predicted_classes, average='weighted')
        f1_score_macro = f1_score(all_targets, predicted_classes, average='macro')
        f1_score_weighted = f1_score(all_targets, predicted_classes, average='weighted')
        
        self.epoch_ema = mean_f1_dice
        overall_auc = roc_auc_score(all_targets, all_predictions, average=None, multi_class='ovr')
        
        if self.best_ema is None or self.epoch_ema > self.best_ema:
            self.best_ema = self.epoch_ema
            self.save_checkpoint(os.path.join(self.output_folder, f"checkpoint_best_Epoch.pth"))
            # Create and save confusion matrix
            num_classes = len(tp)
            conf_matrix = np.vstack([conf_matrix_row0, conf_matrix_row1, conf_matrix_row2])

            class_labels = ["HYPO", "PRL", "HYPER"]
        
            plt.figure(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")

            plt.savefig(f"{self.output_folder}/confusion_matrix.png", dpi=300, bbox_inches="tight")
        
            # Plot ROC curve for each class
            plt.figure(figsize=(8, 6))
            for i in range(3):
                fpr, tpr, _ = roc_curve((all_targets == i).astype(int), all_predictions[:, i])
                plt.plot(fpr, tpr, label=f'{class_labels[i]} (AUC = {roc_auc_score((all_targets == i).astype(int), all_predictions[:, i]):.2f})')
        
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random guessing
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(f"{self.output_folder}/ROC_best_Epoch")
        
        lr = self.optimizer.param_groups[0]["lr"]
        if self.csv_summary is None:
            self.create_summary(["epoch", "train_loss", "val_loss", "F1_per_class", self.ema_measure, "F1 weighted", "Precision macro", "Recall macro", "Precision weighted", "Recall weighted", "AUC", "lr"] + ["tp", "fp", "fn", "tn"])
        
        self.append_metrics_to_summary([
            self.current_epoch, self.mean_epoch_loss, self.mean_val_epoch_loss, f1_per_class, self.epoch_ema, f1_score_weighted, precision_macro, recall_macro, precision_weighted, recall_weighted, overall_auc, lr, *tp, *fp, *fn, *tn
            ])            
