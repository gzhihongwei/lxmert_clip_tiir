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
        return 1.0 if self.img_keys[img_idx] == cap_idx[0] else 0.0

    def prepare_captions(self, text_a, text_b=None):
       tokens_a = self.tokenizer(text_a, return_tensors='pt')
       tokens_a = {k: v.squeeze(0) for k,v in tokens_a.items()}
       return tokens_a

    def __getitem__(self, index):
        outputs = {}
        if self.is_train:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            features = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            example = self.prepare_captions(caption)

            # select a negative pair
            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
            img_idx_neg = random.choice(neg_img_indexs)
            
            label = 1.0
            
            if random.random() <= self.prob_unaligned:
                # When to create an unaligned pair during training
                
                label = 0.0
                
                if random.random() <= 0.5:
                    # randomly select a negative caption from a different image.
                    cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                    caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                    #caption = caption_neg
                    example = self.prepare_captions(caption_neg)
                else:
                    # randomly select a negative image 
                    features = self.get_image(self.img_keys[img_idx_neg])

            outputs['labels'] = label
            outputs.update(features)
            outputs.update(example)
            #outputs['captions'] = caption
            
        else:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            features = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            example = self.prepare_captions(caption)
            label = 1.0 if img_key == cap_idxs[0] else 0.0
            outputs['index'] = index
            outputs['labels'] = label
            outputs.update(features)
            outputs.update(example)
            #outputs['captions'] = caption
            
        return outputs

    def get_image(self, image_id):
        return self.img_feats[str(image_id)]

    def __len__(self):
        if not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        return len(self.img_keys) * self.num_captions_per_img


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin: int = 0, max_violation: bool = False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        
    def _create_similarity_scores(self,
                                  model: torch.nn.Module, 
                                  input_ids=None,
                                  visual_feats=None,
                                  visual_pos=None,
                                  attention_mask=None,
                                  token_type_ids=None,
                                  visual_attention_mask=None) -> torch.tensor:
        all_scores = None
        
        for vis_feat, vis_pos, vis_attn_mask in zip(visual_feats, visual_pos, visual_attention_mask):
            vis_feat = vis_feat.expand(input_ids.size(0), *vis_feat.shape)
            vis_pos = vis_pos.exapnd(input_ids.size(0), *vis_pos.shape)
            vis_attn_mask = vis_attn_mask.expand(input_ids.size(0), *vis_attn_mask.shape)
            scores = model(input_ids=input_ids,
                           visual_feats=vis_feat,
                           visual_pos=vis_pos,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           visual_attention_mask=vis_attn_mask)
            all_scores = scores if not all_scores else torch.cat((all_scores, scores))
        
        return all_scores
            

    def forward(self,
                model: torch.nn.Module, 
                input_ids=None,
                visual_feats=None,
                visual_pos=None,
                attention_mask=None,
                token_type_ids=None,
                visual_attention_mask=None) -> torch.tensor:
        scores = self._create_similarity_scores(model,
                                                input_ids=input_ids,
                                                visual_feats=visual_feats,
                                                visual_pos=visual_pos,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                visual_attention_mask=visual_attention_mask)
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
