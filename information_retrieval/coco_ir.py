# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import json
import random

from typing import Dict, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from information_retrieval.utils import DataTrainingArguments


class RetrievalDataset(Dataset):
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
        super().__init__()
        args.data_path /= split
        caption_file = args.data_path / f"{split}_captions.pt"
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
                with open(args.data_path / args.eval_img_keys_file, "r") as f:
                    img_keys = f.readlines()
                self.img_keys = [k.strip() for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                
        if args.cross_image_eval:
            self.effective_captions_per_img *= len(self.img_keys)
            
        # The probability that a negative pair is sampled
        assert 0 <= args.prob_unaligned < 1, "prob_unaligned must be a probability"
        self.prob_unaligned = args.prob_unaligned
        self.is_train = is_train
        self.tokenizer = tokenizer
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
            self.img_feats = h5py.File(self.args.data_path / f"{self.split}_img_frcnn_feats.h5", "r")
        return dict(**self.img_feats[str(image_id)])

    def __len__(self) -> int:
        if not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        return len(self.img_keys) * self.num_captions_per_img


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin: int = 0, max_violation: bool = False):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation         

    def forward(self, scores: torch.tensor) -> torch.tensor:
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
