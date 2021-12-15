# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

from typing import Dict

import h5py
import numpy as np
import torch

from transformers import CLIPProcessor

from ..utils import DataTrainingArguments
from ..coco import COCORetrievalDataset


class CLIPRetrievalDataset(COCORetrievalDataset):
    """Image/Text Retrieval Dataset"""
    def __init__(self, 
                 processor: CLIPProcessor, 
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
        super().__init__(args=args, split=split, is_train=is_train)

        self.processor = processor

    def process_data(self, caption: str, image: np.ndarray) -> Dict[str, torch.tensor]:
       inputs = self.processor(text=caption, images=image, return_tensors="pt", padding=True)
       inputs = {k: v.squeeze(0) for k, v in inputs.items()}
       return inputs

    def __getitem__(self, index: int) -> Dict[str, any]:
        outputs = {}
        img_idx, cap_idxs = self.get_image_caption_index(index)
        img_key = self.img_keys[img_idx]
        caption = self.captions[cap_idxs[0]][cap_idxs[1]]
        image = self.get_image(img_key)
        outputs = self.process_data(caption=caption, image=image)

        if not self.is_train and self.args.cross_image_eval:
            label = 1.0 if img_key == cap_idxs[0] else 0.0
            outputs['labels'] = label
            
        return outputs

    def get_image(self, image_id: str) -> np.ndarray:
        if not hasattr(self, "images"):
            self.images = h5py.File(self.split_path / f"{self.split}_imgs.h5", "r")
        image = self.images[image_id][()]

        if len(image.shape) == 2:
            image = image[..., np.newaxis].repeat(3, 2)

        return image

