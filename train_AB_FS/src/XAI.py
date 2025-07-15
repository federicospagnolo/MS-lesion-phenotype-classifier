import os, sys
import csv
import pandas as pd
import seaborn as sns
from typing import Any, Union, Dict, List
import numpy as np
import torch
from torchmetrics import ConfusionMatrix
import nibabel as nib
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
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
from src.utils import collate_outputs
from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam, LayerAttribution
from monai.visualize import GradCAMpp, GradCAM
from monai.transforms import SpatialPad
from src.modular_rimnet_mod import ModularRimNet ######
from src.datasets import BATCHKEYS

class BasicRimNetInferer:
    def __init__(
        self, checkpoint_path, model: ModularRimNet, task: str, device=torch.device("cuda")
    ):
        assert isinstance(model, ModularRimNet)
        self.csv_summary = None
        self.device = device
        self.model = model.to(self.device)
        self.task = task
        self.checkpoint = torch.load(checkpoint_path)
        assert self.checkpoint["model_name"] == self.model.__class__.__name__
        self.model.load_state_dict(self.checkpoint["network_weights"])

    def __call__(
        self,
        dataloader,
        test_folder,
        save_keys_in_batch=[BATCHKEYS.SUBJECT, BATCHKEYS.LESION],
    ):
        self.dataloader = dataloader
        timestamp = datetime.now()
        folder_name = f"{timestamp.day}_{timestamp.month}_{timestamp.year}_{timestamp.hour}_{timestamp.minute}_{timestamp.second}"
        self.test_folder = os.path.join(
            test_folder,
            f"exp_{self.model.__class__.__name__}_{folder_name}",
        )
        os.makedirs(self.test_folder, exist_ok=True)
        # Define the path to the CSV file
        csv_file = f"{test_folder}/test_details.csv"
        df = pd.DataFrame(columns=["subject", "lesion", "pred", "target"])
        df.to_csv(csv_file, index=False)
        
        with torch.no_grad():
           self.model.eval()
           outputs = []
              
           for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                
                outputs.append(self.test_step(csv_file, batch))
        
           if self.task == "multiclass":
                self.test_end_multi(outputs)
           else:
                self.test_end(outputs)

    def test_step(self, csv_file, batch: dict) -> dict:
    
        subject = str(batch[BATCHKEYS.SUBJECT][0])
        lesion = str(batch[BATCHKEYS.LESION].tolist()[0])
        
        data = batch[BATCHKEYS.IMAGE].to(self.device, non_blocking=True)  # TODO the imposition of a dataset with this labels and structure
        target = batch[BATCHKEYS.LABEL].to(self.device, non_blocking=True)

        output = self.model(data)
        del data
        
        predict_proba = torch.softmax(output, dim=1)
        output_binary = torch.argmax(predict_proba, dim=1)
        
        # Convert output and target to binary decisions (0 or 1)
        output_binary = output_binary.cpu().tolist()
        target_binary = target.detach().cpu().tolist()

        #########
        print_test = True
        ######### 
        
        # Iterate through the batch to determine the result type for each sample
        if print_test == True:
        
          # Prepare a list to store results
          results = []

          # Iterate through the batch to determine the result type for each sample
          for i in range(len(output_binary)):
            subject = str(batch[BATCHKEYS.SUBJECT][i])  # Extract subject
            lesion = str(batch[BATCHKEYS.LESION].tolist()[i])  # Extract lesion

            # Determine the result type based on the conditions
            if output_binary[i] != target_binary[i]:

                # Append the result for misclassified example
                results.append([subject, lesion, output_binary[i], target_binary[i]])

          # Convert the results to a DataFrame and append to the CSV file
          df = pd.DataFrame(results)
          df.to_csv(csv_file, mode='a', header=False, index=False)

        if self.task == "multiclass":
            return {
            **self.compute_test_metrics_multi(target, output),
        }
        else:
            return {
            **self.compute_test_metrics(target, output),
        }
        
    def test_end(self, outputs: List[dict]):
        outputs_collated = collate_outputs(outputs)
        tp = np.sum(outputs_collated["tp"])
        fp = np.sum(outputs_collated["fp"])
        fn = np.sum(outputs_collated["fn"])
        tn = np.sum(outputs_collated["tn"])

        global_f1_per_class = [
            i for i in [2 * i / (2 * i + j + k) for i, j, k in zip([tp], [fp], [fn])]
        ]
        mean_f1_dice = np.nanmean(global_f1_per_class)

        # Accumulate all predictions and targets for AUC calculation
        all_predictions = np.concatenate([pred.flatten() for pred in outputs_collated["predictions"]])
        all_predictions_bin = (all_predictions >= 0.5).astype(int)
        all_targets = np.concatenate([pred.flatten() for pred in outputs_collated["targets"]])
        # Calculate overall AUC on the entire validation set
        overall_auc = roc_auc_score(all_targets, all_predictions)
        # Precision and recall
        precision = precision_score(all_targets, all_predictions_bin)
        recall = recall_score(all_targets, all_predictions_bin)
        
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
        plt.savefig(f"{self.test_folder}/ROC")
        
        if self.csv_summary is None:
            self.create_summary(
                ['F1', 'AUC', 'precision', 'recall', 'tp_t', 'fp_t', 'fn_t', 'tn_t']
            )
            
        self.append_metrics_to_summary(
            [mean_f1_dice, overall_auc, precision, recall, tp, fp, fn, tn]
        )        
        
    def compute_test_metrics(self, target, model_output) -> Dict:
        
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
        #monoclass_batch = len(np.unique(target)) == 1
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
        
    def create_summary(self, columns):
        self.csv_summary = os.path.join(self.test_folder, "summary.csv")
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
            
    def test_end_multi(self, outputs: List[dict]):
        outputs_collated = collate_outputs(outputs)
        
        conf_matrix_row0 = np.array(outputs_collated["conf_matrix_row0"]).reshape(-1, 3).sum(axis=0)
        conf_matrix_row1 = np.array(outputs_collated["conf_matrix_row1"]).reshape(-1, 3).sum(axis=0)
        conf_matrix_row2 = np.array(outputs_collated["conf_matrix_row2"]).reshape(-1, 3).sum(axis=0)
        
        tp = np.array(outputs_collated["tp"]).reshape(-1, 3).sum(axis=0)
        fp = np.array(outputs_collated["fp"]).reshape(-1, 3).sum(axis=0)
        fn = np.array(outputs_collated["fn"]).reshape(-1, 3).sum(axis=0)
        tn = np.array(outputs_collated["tn"]).reshape(-1, 3).sum(axis=0)

        all_predictions = outputs_collated["predictions"]
        all_targets = outputs_collated["targets"]
        
        f1_per_class = [2 * tp[i] / (2 * tp[i] + fp[i] + fn[i]) if (2 * tp[i] + fp[i] + fn[i]) > 0 else 0 for i in range(len(tp))]
        f1_score_macro = np.nanmean(f1_per_class)
        
        predicted_classes = np.argmax(all_predictions, axis=1)
        precision_macro = precision_score(all_targets, predicted_classes, average='macro')
        recall_macro = recall_score(all_targets, predicted_classes, average='macro')
        precision_weighted = precision_score(all_targets, predicted_classes, average='weighted')
        recall_weighted = recall_score(all_targets, predicted_classes, average='weighted')
        f1_score_weighted = f1_score(all_targets, predicted_classes, average='weighted')
        
        num_classes = len(tp)
        conf_matrix = np.vstack([conf_matrix_row0, conf_matrix_row1, conf_matrix_row2])

        class_labels = ["HYPO", "PRL", "HYPER"]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        plt.savefig(f"{self.test_folder}/confusion_matrix.png", dpi=300, bbox_inches="tight")
        
        # Calculate overall AUC on the entire test set
        overall_auc = roc_auc_score(all_targets, all_predictions, average=None, multi_class='ovr')
        
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
        plt.savefig(f"{self.test_folder}/ROC_best_Epoch")
        
        if self.csv_summary is None:
            self.create_summary(
                ["F1_per_class", "F1", "F1 weighted", "Precision macro", "Recall macro", "Precision weighted", "Recall weighted", "AUC", "tp_t", "fp_t", "fn_t", "tn_t"]
            )
            
        self.append_metrics_to_summary(
            [f1_per_class, f1_score_macro, f1_score_weighted, precision_macro, recall_macro, precision_weighted, recall_weighted, overall_auc, *tp, *fp, *fn, *tn]
        )        
        
    def compute_test_metrics_multi(self, target, model_output) -> Dict:
        num_classes = 3  # Adjusted for multi-class classification
        
        model_output = model_output.view(-1, num_classes)  # Reshaping to (N, num_classes)

        # Apply softmax to get probabilities for both classes
        predict_proba = torch.softmax(model_output, dim=1)
        
        # Convert to binary output (take the index of the class with the highest probability)
        binary_outputs = torch.argmax(predict_proba, dim=1)
        
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

class BasicRimNetXAI:
    def __init__(
        self, checkpoint_path, model: ModularRimNet, task: str, device=torch.device("cuda")
    ):
        assert isinstance(model, ModularRimNet)
        self.device = device
        self.model = model.to(self.device)
        self.checkpoint = torch.load(checkpoint_path)
        assert self.checkpoint["model_name"] == self.model.__class__.__name__
        self.model.load_state_dict(self.checkpoint["network_weights"])

    def __call__(
        self,
        dataloader,
        test_folder,
        save_keys_in_batch=[BATCHKEYS.SUBJECT, BATCHKEYS.LESION],
    ):
        self.xai_folder = os.path.join(
            test_folder,
            f"XAI",
        )
        os.makedirs(self.xai_folder, exist_ok=True)

        predictions_df = []
        self.model.eval()
        # instatiate IG
        ig = IntegratedGradients(self.model) # default number of steps used by the approximation method: 50.
        # SG parameters
        std_fraction = 0.05
        nb_smooth = 50
        for name, _ in self.model.named_modules(): print(name)
        for batch in tqdm(dataloader):
                subject = str(batch[BATCHKEYS.SUBJECT][0])
                lesion = str(batch[BATCHKEYS.LESION].tolist()[0])
                data = batch[BATCHKEYS.IMAGE].to(self.device)
                std = std_fraction * (data[0,:].max() - data[0,:].min())
                #print(f"Subject {subject}, lesion: {lesion}")
                output = self.model(data)
                predict_proba = torch.softmax(output, dim=1)
                pred = predict_proba[0,1].item()

                prediction_dict = {
                    **{k: batch[k] for k in save_keys_in_batch},
                    **{
                        cls: predict_proba[:, cls].detach().cpu().numpy()
                        for cls in range(predict_proba.shape[-1])
                    },
                }  # The model outpurt should be always [BS, n_classes]

                predictions_df.append(pd.DataFrame(prediction_dict))
                
                #grads = torch.zeros_like(data, device=self.device)
                #for q in range(nb_smooth):
                #    noisy_input = data + data.new(data.size()).normal_(0, std)
                #    noisy_input.requires_grad_()
                #    output = self.model(noisy_input)
                #    predict_proba = torch.softmax(output, dim=1).squeeze(0)
                #    predict_proba[1].backward() # selecting class 1 (?)
                #    grads = grads + noisy_input.grad
            
                #grads = (grads.squeeze(0) / (nb_smooth)).detach().cpu().numpy()
                
                data = batch[BATCHKEYS.IMAGE].numpy()
                data = torch.from_numpy(data).to(self.device)
                
                # define baseline for IG
                #baselines = data + data.new(data.size()).normal_(0, std)
                #baselines = -torch.ones_like(data) # default baseline is ones tensors
                baselines = torch.zeros_like(data) # default baseline is zeros tensors
                integ = ig.attribute(data, baselines=baselines, target=1).cpu().numpy().squeeze()
                
                # instantiate GCAM
                #layer = "subnet_for_modality.0.2.1"
                flair_layers = [self.model.subnet_for_modality[1][2][0].conv_block[0], self.model.subnet_for_modality[1][2][0].conv_block[3], self.model.subnet_for_modality[1][1][0].conv_block[3]]
                phase_layers = [self.model.subnet_for_modality[2][2][0].conv_block[0], self.model.subnet_for_modality[2][2][0].conv_block[3], self.model.subnet_for_modality[2][1][0].conv_block[3]]
                #flair_layers = [self.model.subnet_for_modality[0][0][0].conv_block[3], self.model.subnet_for_modality[0][0][0].conv_block[5], self.model.subnet_for_modality[0][1][0].conv_block[1], self.model.subnet_for_modality[0][1][0].conv_block[3], self.model.subnet_for_modality[0][1][0].conv_block[5], self.model.subnet_for_modality[0][2][0].conv_block[5]]
                #phase_layers = [self.model.subnet_for_modality[1][0][0].conv_block[3], self.model.subnet_for_modality[1][0][0].conv_block[5], self.model.subnet_for_modality[1][1][0].conv_block[1], self.model.subnet_for_modality[1][1][0].conv_block[3], self.model.subnet_for_modality[1][1][0].conv_block[5], self.model.subnet_for_modality[1][2][0].conv_block[5]]
                #cam = GradCAMpp(nn_module=self.model, target_layers=layer) # monai
                
                # GCAM captum mean across layers
                flair_gcam = torch.zeros_like(data[0,0], device='cpu').numpy()
                #print(flair_gcam.shape)
                for layer in flair_layers:
                    cam = LayerGradCam(self.model, layer)
                    gcam_raw = cam.attribute(data, 1)
                    #print(gcam_raw.size())
                    gcam = LayerAttribution.interpolate(gcam_raw, (28, 28, 28), interpolate_mode='nearest').detach().cpu().numpy().squeeze().squeeze()
                    gcam = (gcam - np.min(gcam)) / (np.max(gcam) - np.min(gcam))
                    flair_gcam = flair_gcam + gcam
                flair_gcam = flair_gcam / len(flair_layers)
                
                phase_gcam = torch.zeros_like(data[0,0], device='cpu').numpy()
                for layer in phase_layers:
                    cam = LayerGradCam(self.model, layer)
                    gcam_raw = cam.attribute(data, 1)
                    #print(gcam_raw.size())
                    gcam = LayerAttribution.interpolate(gcam_raw, (28, 28, 28), interpolate_mode='nearest').detach().cpu().numpy().squeeze().squeeze()
                    gcam = (gcam - np.min(gcam)) / (np.max(gcam) - np.min(gcam))
                    phase_gcam = phase_gcam + gcam
                phase_gcam = phase_gcam / len(phase_layers)  
                
                data = data.detach().cpu().numpy().squeeze()
                baselines = baselines.detach().cpu().numpy().squeeze()
            
                tag = f'{subject}_{lesion}_T1.nii.gz'
                image_path = os.path.join(
                test_folder,
                tag)

                input_affine = nib.load(image_path).affine
                #g_fl = nib.Nifti1Image(flair_gcam[0], input_affine) # grads wrt flair
                #g_ph = nib.Nifti1Image(phase_gcam[1], input_affine) # grads wrt phase
                #flair = nib.Nifti1Image(data[0], input_affine)
                #phase = nib.Nifti1Image(data[1], input_affine)
                gcam_fl = nib.Nifti1Image(flair_gcam, input_affine)
                gcam_ph = nib.Nifti1Image(phase_gcam, input_affine)
                #gcam_flair = nib.Nifti1Image(gcam0, input_affine)
                #gcam_phase = nib.Nifti1Image(gcam1, input_affine)
                baseline = nib.Nifti1Image(baselines[0], input_affine)
                ig_flair = nib.Nifti1Image(integ[1], input_affine)
                ig_phase = nib.Nifti1Image(integ[2], input_affine)
                
                # sort in folders
                if pred > 0.5:
                     if int(batch[BATCHKEYS.LESION].item()) < 2000 or (int(batch[BATCHKEYS.LESION].item()) < 8000 and int(batch[BATCHKEYS.LESION].item()) >= 7000):
                          group = 'TP'
                     else:
                          continue
                          group = 'FP'
                else:
                     continue
                     if int(batch[BATCHKEYS.LESION].item()) < 2000:
                          group = 'FN'
                     else:
                          group = 'TN'        

                #nib.save(g_ph, self.xai_folder + "/" + group + "/grads_ph_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                #nib.save(g_fl, self.xai_folder + "/" + group + "/grads_fl_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                #nib.save(flair, self.xai_folder + "/" + group + "/flair_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                #nib.save(phase, self.xai_folder + "/" + group + "/phase_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                #nib.save(gcam, self.xai_folder + "/" + group + "/GCAM_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                #nib.save(gcam_fl, self.xai_folder + "/" + group + "/GCAM_fl_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                #nib.save(gcam_ph, self.xai_folder + "/" + group + "/GCAM_ph_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                #nib.save(baseline, self.xai_folder + "/" + group + "/baseline_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                #nib.save(ig_flair, self.xai_folder + "/" + group + "/IG_fl_" + subject +
                #"_Lesion_" + lesion + ".nii.gz")
                os.makedirs(self.xai_folder + "/" + group, exist_ok=True)
                nib.save(ig_phase, self.xai_folder + "/" + group + "/IG_ph_" + subject +
                "_Lesion_" + lesion + ".nii.gz")
               
        pd.concat(predictions_df).to_csv(self.xai_folder + '/results.csv')        
