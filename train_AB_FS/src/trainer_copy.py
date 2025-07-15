from typing import Any, Union, Dict, List

import sys
import os
import csv
from time import time
from datetime import datetime
from tqdm import tqdm, trange

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    auc,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from src.modular_rimnet import ModularRimNet
from src.loggers import AbstractLogger, FileLogger
from src.utils import empty_cache, collate_outputs, flatten_nested_numbers
from src.SMSC import BATCHKEYS


class BasicRimNetTrainer:
    def __init__(
        self,
        model: ModularRimNet,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_path: str,
        loss_fn=None,
        optimizer=None,
        lr_scheduler=None,
        num_epochs: int = 50,
        num_iterations_per_epoch: int = None,
        num_val_iterations_per_epoch: int = None,
        save_every_epochs=10,
        device: torch.device = torch.device("cuda"),
        logger: AbstractLogger = None,
        ema_measure="F1",
    ):
        # dataloader for training and validation
        assert isinstance(train_dataloader, DataLoader)
        assert isinstance(val_dataloader, DataLoader)
        assert isinstance(
            model, ModularRimNet
        ), f"The model: {model} is not a ModularRimNet"  # this is important for the input spliting
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
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

            # Validation block
            with torch.no_grad():
                self.model.eval()
                val_outputs = {}
                for batch_idx, batch in enumerate(self.val_dataloader):
                    if batch_idx >= self.num_val_iterations_per_epoch:
                        continue
                    val_output = self.validation_step(batch)
                    val_outputs[f'batch_{batch_idx}'] = val_output

                self.validation_epoch_end(val_outputs)
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

    def validation_step(self, batch: dict) -> dict:
        data = batch[
            BATCHKEYS.IMAGE
        ]  # TODO the imposition of a dataset with this labels and structure
        target = batch[BATCHKEYS.LABEL]

        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        output = self.model(data)
        del data
        l = self.loss(output, target)
        self.step_val_loss = l.clone()

        predict_proba, binary_outputs = torch.max(
            torch.softmax(output, dim=1), dim=1)

        return {
            **{"loss": np.array([l.detach().cpu().numpy()])},
            **{"target": np.array(target.detach().cpu().numpy())},
            **{"output": np.array(binary_outputs.detach().cpu().numpy())},
            **{"probs": np.array(predict_proba.detach().cpu().numpy())}

        }

    # This is the definition for LRPlateau. Extend the class for a different Scheduler
    def scheduler_step(self):
        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(self.mean_epoch_loss)
        else:
            self.lr_scheduler.step()

    def validation_epoch_end(self, val_outputs: Dict[str, dict]):
        
        targets = np.concatenate([val_output["target"] for val_output in val_outputs.values()]).ravel()
        outputs = np.concatenate([val_output["output"] for val_output in val_outputs.values()]).ravel()
        probs = np.concatenate([val_output["probs"] for val_output in val_outputs.values()]).ravel()
        val_loss = np.mean(np.concatenate([val_output["loss"] for val_output in val_outputs.values()]).ravel())
        self.mean_val_epoch_loss = val_loss
        
        cm = confusion_matrix(targets, outputs)
        tn, fp, fn, tp = cm.ravel()

        global_f1_per_class = [
            i for i in [2 * i / (2 * i + j + k) for i, j, k in zip([tp], [fp], [fn])]
        ]
        mean_f1_dice = np.nanmean(global_f1_per_class)
        self.epoch_ema = mean_f1_dice
        overall_auc = roc_auc_score(targets, probs)
        
        if self.best_ema is None or self.epoch_ema > self.best_ema:
            self.best_ema = self.epoch_ema
            # print(f"New best EMA {self.ema_measure} : {np.round(self.best_ema, decimals=4)}")
            self.save_checkpoint(
                os.path.join(self.output_folder, f"checkpoint_best_epoch{self.current_epoch}.pth")
            )

        # Since the scheduler could be None, its safer to acces lr this way
        lr = self.optimizer.param_groups[0]["lr"]
        if self.csv_summary is None:
            self.create_summary(
                ["epoch", "train_loss", 'val_loss', self.ema_measure, "AUC", 'TP', 'FP', 'FN', 'TN',"lr"]
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
