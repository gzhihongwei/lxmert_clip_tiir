
# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import json

from pathlib import Path
from typing import Dict, Tuple

import torch

from torch.utils.data import Dataset

from .utils import DataTrainingArguments


class COCORetrievalDataset(Dataset):
    """Image/Text Retrieval Dataset"""
    def __init__(self, 
                 args: DataTrainingArguments,
                 split: str = 'train',
                 is_train: bool = True):
        """
        tokenizer: tokenizer to process caption text.
        args: configuration parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super().__init__()
        self.split_path = Path(args.data_path) / split
        caption_file = self.split_path / f"{split}_captions.pt"
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        # There are 5 captions per image in COCO
        self.num_captions_per_img = self.effective_captions_per_img = 5
        
        if not is_train:
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(self.split_path / args.eval_img_keys_file, "r") as f:
                    img_keys = f.readlines()
                self.img_keys = [k.strip() for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                
        if args.cross_image_eval:
            self.effective_captions_per_img *= len(self.img_keys)
            
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

    def __getitem__(self, index: int) -> Dict[str, any]:
        pass

    def __len__(self) -> int:
        if not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        return len(self.img_keys) * self.num_captions_per_img

