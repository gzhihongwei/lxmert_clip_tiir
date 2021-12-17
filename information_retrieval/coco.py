from __future__ import absolute_import, division

import json

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import torch

from torch.utils.data import Dataset

from .utils import DataTrainingArguments


class COCORetrievalDataset(Dataset, ABC):
    """Base image-text retrieval PyTorch Dataset that wraps MSCOCO."""
    
    def __init__(self, 
                 args: DataTrainingArguments,
                 split: str = 'train',
                 is_train: bool = True):
        """
        args: configuration parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
            Images are in the `.h5` format, which allows the file to be read into RAM while lazily loading
            each "dataset," or image feature/image in HDF5. Captions are in `.pt` files, which contains a
            JSON string of all 5 captions per caption, which is indexed by the corresponding image ID given
            in MSCOCO.
        is_train
        """
        
        # Initialize PyTorch Dataset superclass
        super().__init__()
        
        # Load the captions in
        self.split_path = Path(args.data_path) / split
        caption_file = self.split_path / f"{split}_captions.pt"
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())
        
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        # There are 5 captions per image in COCO
        self.num_captions_per_img = 5
        
        if not is_train:
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(self.split_path / args.eval_img_keys_file, "r") as f:
                    img_keys = f.readlines()
                self.img_keys = [k.strip() for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                
            if args.cross_image_eval:
                self.num_captions_per_img *= len(self.img_keys)
            
        # The probability that a negative pair is sampled
        self.split = split
        self.is_train = is_train
        self.args = args

    def get_image_caption_index(self, index: int) -> Tuple[int, Tuple[int, int]]:
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))
            img_idx1 = cap_idx // self.num_captions_per_img
            cap_idx1 = cap_idx % self.num_captions_per_img
            return img_idx, (self.img_keys[img_idx1], cap_idx1)
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        return img_idx, (self.img_keys[img_idx], cap_idx)

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, any]:
        pass

    def __len__(self) -> int:
        return len(self.img_keys) * self.num_captions_per_img
