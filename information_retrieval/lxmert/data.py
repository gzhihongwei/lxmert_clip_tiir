# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import random

from typing import Dict, Union

import h5py
import numpy as np
import torch

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..utils import DataTrainingArguments
from ..coco import COCORetrievalDataset


class LxmertRetrievalDataset(COCORetrievalDataset):
    """Image/Text Retrieval Dataset"""
    def __init__(self, 
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
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
        
        # The probability that a negative pair is sampled
        assert 0 <= args.prob_unaligned < 1, "prob_unaligned must be a probability"
        self.prob_unaligned = args.prob_unaligned

        self.tokenizer = tokenizer

    def prepare_caption(self, caption: str) -> Dict[str, torch.tensor]:
       caption = self.tokenizer(caption, return_tensors='pt')
       caption = {k: v.squeeze(0) for k,v in caption.items()}
       return caption

    def __getitem__(self, index: int) -> Dict[str, any]:
        outputs = {}
        if self.is_train:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            img_feats = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            textual_data = self.prepare_caption(caption)

            # select a negative pair
            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
            img_idx_neg = random.choice(neg_img_indexs)
            
            label = 1.0
            
            if random.random() < self.prob_unaligned:
                # When to create an unaligned pair during training
                
                label = 0.0
                
                if random.random() < 0.5:
                    # randomly select a negative caption from a different image.
                    cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                    caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                    textual_data = self.prepare_caption(caption_neg)
                else:
                    # randomly select a negative image 
                    img_feats = self.get_image(self.img_keys[img_idx_neg])

            outputs['labels'] = label
            outputs.update(img_feats)
            outputs.update(textual_data)
            
        else:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            img_feats = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            textual_data = self.prepare_caption(caption)
            label = 1.0 if img_key == cap_idxs[0] else 0.0
            outputs['labels'] = label

            outputs.update(img_feats)
            outputs.update(textual_data)
            
        return outputs

    def get_image(self, image_id: int) -> Dict[str, np.ndarray]:
        if not hasattr(self, "img_feats"):
            self.img_feats = h5py.File(self.split_path / f"{self.split}_img_frcnn_feats.h5", "r")
        return dict(**self.img_feats[str(image_id)])
