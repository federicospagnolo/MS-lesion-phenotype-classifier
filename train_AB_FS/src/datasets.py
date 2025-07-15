import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from monai.transforms import LoadImage
from typing import Union, List, Tuple, Dict


class CSVINFO:
    LABEL = "Label"
    SUBJECT = "Subject"
    LESION = "Lesion_id"


class BATCHKEYS:
    IMAGE = "img"
    SUBJECT = "subject"
    LESION = "lesion"
    LABEL = "label"


class PatchesFromCSV(Dataset):
    def __init__(
        self,
        csv_path,
        use_modalities: List[str],
        label_id=CSVINFO.LABEL,
        subject_id=CSVINFO.SUBJECT,
        lesion_id=CSVINFO.LESION,
        keep_extra_cols: List[str] = None,
        relative_paths=True,
        transform=None,
    ):
        self.csv = csv_path
        self.df = pd.read_csv(csv_path)
        self.deployment_dataset = False
        if label_id is not None:
            assert (
                label_id in self.df.columns
            ), f"The specified label_id column: {label_id} is not presensent in the CSV"
            self.label_id = label_id
        else:
            print("Not Label provided. Considering deployment dataset")
            self.self.deployment_dataset = True
        assert (
            subject_id in self.df.columns
        ), f"The specified subject_id column: {subject_id} is not presensent in the CSV"
        self.subject_id = subject_id
        assert (
            lesion_id in self.df.columns
        ), f"The specified lesion_id column: {lesion_id} is not presensent in the CSV"
        self.lesion_id = lesion_id
        self.modalities = use_modalities
        print(
            f"Proposed Modalities {use_modalities}. Modalities in the CSV to be employed {self.modalities}"
        )
        self.loader = LoadImage(ensure_channel_first=True)
        self.imgs_dir = os.path.dirname(self.csv) if relative_paths else ""
        self.keep_extra_cols = keep_extra_cols
        self.transform = transform

    @property
    def modalities(self) -> List[str]:
        return self._modalities

    @modalities.setter
    def modalities(self, modalities):
        modalities = self._which_columns_exits(modalities)
        assert (
            len(modalities) > 0
        ), "None of the given modalities is present in the CSV file"
        self._modalities = modalities

    @property
    def keep_extra_cols(self):
        return self._keep_extra_cols

    @keep_extra_cols.setter
    def keep_extra_cols(self, keep_extra_cols):
        self._keep_extra_cols = (
            []
            if keep_extra_cols is None
            else self._which_columns_exits(keep_extra_cols)
        )

    def _which_columns_exits(self, columns_names):
        # do not use set since are unsorted
        assert isinstance(columns_names, list)
        return [el for el in columns_names if el in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index): # The apply is just a workaround
        # Load and concatenate image modalities
        img = torch.cat(
            [
                self.loader(os.path.join(self.imgs_dir, self.df[modality][index]))
                for modality in self.modalities
            ]
        )
        if self.label_id:
            drawn = {
                **{
                    BATCHKEYS.IMAGE: img,
                    BATCHKEYS.SUBJECT: self.df[self.subject_id][index],
                    BATCHKEYS.LESION: self.df[self.lesion_id][index],
                    BATCHKEYS.LABEL: self.df[self.label_id][index],
                },
                **{
                    extra_col: self.df[extra_col][index]
                    for extra_col in self.keep_extra_cols
                },
            }
        else:
            drawn = {
                **{
                    BATCHKEYS.IMAGE: img,
                    BATCHKEYS.SUBJECT: self.df[self.subject_id][index],
                    BATCHKEYS.LESION: self.df[self.lesion_id][index],
                },
                **{
                    extra_col: self.df[extra_col][index]
                    for extra_col in self.keep_extra_cols
                },
            }
        # Apply transforms
        if self.transform:
            drawn = self.transform(drawn)

        return drawn

    def get_labels(self):
        return self.df.Label  

class PatchesFromCSVCached(PatchesFromCSV):
    def __init__(self, own_transform = None,*args,**kwargs):
        super(PatchesFromCSVCached, self).__init__(*args,**kwargs)
        self.own_transform = own_transform
        self.cached_items = [None] * super().__len__()
    def __getitem__(self, index):
        if self.cached_items[index]:
            #print(f'The element {index} is already cached')
            item = self.cached_items[index]
        else: 
            print(f'The element {index} to be cached')
            item = super().__getitem__(index)
            self.cached_items[index] = item

        if self.own_transform:
            #print('transform_own')
            item = self.own_transform(item)

        return item
    
            
