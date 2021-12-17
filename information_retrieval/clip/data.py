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
        Initialize the CLIP version of the retrieval Dataset.

        Args:
            processor (CLIPProcessor): The HuggingFace CLIPProcessor.
            args (DataTrainingArguments): Configuration arguments.
            split (str, optional): What split to return a Dataset for. Defaults to 'train'.
            is_train (bool, optional): Whether training or not. Defaults to True.
        """
        super().__init__(args=args, split=split, is_train=is_train)

        self.processor = processor

    def process_data(self, caption: str, image: np.ndarray) -> Dict[str, torch.tensor]:
        """
        Process the caption and image with the CLIPProcessor

        Args:
            caption (str): Caption string from COCO.
            image (np.ndarray): Image from the `.h5` file.

        Returns:
            Dict[str, torch.tensor]: Prepared dictionary that can be unpacked to kwargs for the model.
        """
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def __getitem__(self, index: int) -> Dict[str, any]:
        """
        Prepares dictionary of kwargs for corresponding model forward.

        Args:
            index (int): What to index.

        Returns:
            Dict[str, any]: Dictionary that can be unpacked to model forward kwargs.
        """
        
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
        """
        Get the image for the given image id.

        Args:
            image_id (str): Image id in MSCOCO.

        Returns:
            np.ndarray: Image with the given image id.
        """
        
        # This cannot be done in __init__ because then with multiple workers, it doesn't work
        if not hasattr(self, "images"):
            self.images = h5py.File(self.split_path / f"{self.split}_imgs.h5", "r")
            
        # Fully get the image
        image = self.images[image_id][()]

        # Greyscale images
        if len(image.shape) == 2:
            image = image[..., np.newaxis].repeat(3, 2)

        return image

