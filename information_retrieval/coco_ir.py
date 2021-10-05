# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import json
import os
import random

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tqdm import tqdm


class RetrievalDataset(Dataset):
    """Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, split='train', is_train=True):
        """
        tokenizer: tokenizer to process caption text.
        args: configuration parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(RetrievalDataset, self).__init__()
        caption_file = os.path.join(args.data_path, split, '{}_captions.pt'.format(split))
        img_file = os.path.join(args.data_path, split, '{}_img_frcnn_feats.pt'.format(split))
        self.img_feats = torch.load(img_file)
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        # There are 5 captions per image in COCO
        self.num_captions_per_img = 5
        
        if not is_train:
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(os.path.join(args.data_path, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                self.img_feats = {k: self.img_feats[k] for k in self.img_keys}

            if args.eval_caption_index_file:
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indexs = True
                assert not args.cross_image_eval 
                caption_index_file = os.path.join(args.data_path, args.eval_caption_index_file)
                self.caption_indexs = torch.load(caption_index_file)
                if not type(self.caption_indexs[self.img_keys[0]]) == list:
                    self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}
            else:
                self.has_caption_indexs = False
                
        # The probability that a negative pair is sampled
        assert 0 <= args.prob_unaligned <= 1, "prob_unaligned must be a probability"
        self.prob_unaligned = args.prob_unaligned
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.args = args

    def get_image_caption_index(self, index):
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))
            img_idx1 = cap_idx // self.num_captions_per_img
            cap_idx1 = cap_idx % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx1], cap_idx1]
        if not self.is_train and self.has_caption_indexs:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
            return img_idx, [img_key1, cap_idx1]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        return img_idx, [self.img_keys[img_idx], cap_idx]

    def get_label(self, index):
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0

    def tensorize_text(self, text_a, text_b=None):
        tokens_a = self.tokenizer(text_a, padding='max_length', return_tensors='pt')
        tokens_a = {k: v.squeeze(0) for k,v in tokens_a.items()}
        return tokens_a

    def __getitem__(self, index):
        outputs = {}
        if self.is_train:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            features = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            example = self.tensorize_text(caption)

            # select a negative pair
            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
            img_idx_neg = random.choice(neg_img_indexs)
            
            label = 1
            
            if random.random() <= self.prob_unaligned:
                # When to create an unaligned pair during training
                
                label = 0
                
                if random.random() <= 0.5:
                    # randomly select a negative caption from a different image.
                    cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                    caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                    example = self.tensorize_text(caption_neg)
                else:
                    # randomly select a negative image 
                    features = self.get_image(self.img_keys[img_idx_neg])

            outputs['labels'] = label
            outputs.update(features)
            outputs.update(example)
            
        else:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            features = self.get_image(img_key)
            captions = self.captions[cap_idxs[0]][cap_idxs[1]]
            tokenized = self.tensorize_text(captions)
            label = 1 if img_key == cap_idxs[0] else 0
            outputs['index'] = index
            outputs['labels'] = label
            outputs.update(features)
            outputs.update(tokenized)
            
        return outputs

    def get_image(self, image_id):
        return self.img_feats[str(image_id)]

    def __len__(self):
        if not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        return len(self.img_keys) * self.num_captions_per_img


def compute_score_with_logits(logits, labels):
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data # argmax
        scores = logits == labels 
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores
