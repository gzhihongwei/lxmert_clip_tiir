import numpy as np
import os
import random

import torch
import torch.utils.data as data
from transformers.models.lxmert.tokenization_lxmert_fast import LxmertTokenizerFast


def get_paths(path, use_restval=False):
    """
    Returns paths to images and annotations for the MSCOCO dataset. The indices 
    are extracted from the Karpathy et al. splits. Special thanks to Faghri for
    the code.
    
    :param path: path to MSCOCO root
    :param use_restval: if True, then the `restval` data is included in train.
    :return: dictionary of paths and dictionary of the caption ids
    """
    
    # Stores the image roots and annotation file
    roots = {}
    # Stores the caption ids
    ids = {}
    
    # Path to the Faster RCNN extracted features
    faster_dir = os.path.join(path, 'faster_features')
    # Path to the annotations directory
    capdir = os.path.join(path, 'annotations')
    
    # Paths for the regular training split
    roots['train'] = {
        'img': os.path.join(faster_dir, 'train2014'),
        'cap': os.path.join(capdir, 'captions_train2014.json')
    }
    
    # Paths for the regular validation split
    roots['val'] = {
        'img': os.path.join(faster_dir, 'val2014'),
        'cap': os.path.join(capdir, 'captions_val2014.json')
    }
    
    # Paths for the regular test split
    roots['test'] = {
        'img': os.path.join(faster_dir, 'val2014'),
        'cap': os.path.join(capdir, 'captions_val2014.json')
    }
    
    # Paths for the 'restval' split defined in Karpathy et al.
    roots['trainrestval'] = {
        'img': (roots['train']['img'], roots['val']['img']),
        'cap': (roots['train']['cap'], roots['val']['cap'])
    }
    
    # Caption ids for each split
    ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
    ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
    ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
    # Combines 'restval' caption ids with that of train
    ids['trainrestval'] = (
        ids['train'],
        np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
    
    # Should training split include 'restval'?
    if use_restval:
        roots['train'] = roots['trainrestval']
        ids['train'] = ids['trainrestval']
        
    return roots, ids


def get_dataset(root, json, prob_unaligned, ids=None, testing=False):
    """
    Returns torch.utils.data.DataSet for custom COCO dataset.
    
    :param root: path to coco directory
    :param json: path to annotation file
    :param prob_unaligned: probability that the image-caption is aligned
    :param ids: caption ids to consider
    :param testing: whether the split in the dataset is for evaluation
    :return: torch.utils.data.DataSet of the specified split of COCO
    """
    # COCO custom dataset
    dataset = CoCoDataset(root=root,
                          json=json,
                          prob_unaligned=prob_unaligned,
                          ids=ids,
                          testing=testing)
    
    return dataset

def get_train_dataset(opt):
    """
    Returns the training PyTorch DataSet for COCO.
    
    :param opt: parsed arguments from argparse
    :return: torch.utils.data.DataSet for the training split (might include
             'restval' based on opt)
    """
    
    # Path to coco directory
    dpath = os.path.join(opt.data_path, 'coco')
    # Build the paths to everything
    roots, ids = get_paths(dpath, opt.use_restval)

    # Get the dataset
    train_dataset = get_dataset(roots['train']['img'],
                                roots['train']['cap'],
                                opt.prob_unaligned,
                                ids=ids['train'])

    return train_dataset


def get_test_dataset(split_name, opt):
    """
    Returns the test PyTorch DataSet for COCO.
    
    :param split_name: name of split, either 'val' or 'test'
    :param opt: parsed arguments from argparse
    :return: torch.utils.data.DataSet for the split specified according to Karpathy et al.y
    """
    
    # Sanity check
    assert split_name in {'val', 'test'}, 'Evaluation split must be either validation or test'
    
    # Path to coco directory
    dpath = os.path.join(opt.data_path, 'coco')
    # Build the paths to everything
    roots, ids = get_paths(dpath, opt.use_restval)

    # Get the loader
    test_dataset = get_dataset(roots[split_name]['img'],
                               roots[split_name]['cap'],
                               0, ids=ids[split_name],
                               testing=True)
    return test_dataset


class CoCoDataset(data.Dataset):
    """
    Pytorch dataset that handles COCO train2014 and val2014 splits
    """
    def __init__(self, root, json, prob_unaligned, ids, testing=False):
        """
        Constructor for CoCoDataset
        
        :param feature_root: directory where all of the saved Faster RCNN features are located
        :param ann_file: annotation file for the data to be loaded
        :param prob_unaligned: probability that the Faster RCNN feature is not from the image
                               paired with the caption
        """
        from pycocotools.coco import COCO
        
        # Store the root
        self.root = root
        
        # When using the 'restval' split, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
         
        # Caption ids from get_paths   
        self.ids = ids
        
        # If 'restval' data is used, record the break point for the ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        
        # Probability of returning an unaligned image from caption
        assert 0 <= prob_unaligned <= 1, "prob_unaligned must be a probability" 
        self.prob_unaligned = prob_unaligned
        
        # Keeps track of whether the dataset is for evaluation or testing
        self.testing = testing
        
        self.tokenizer = LxmertTokenizerFast.from_pretrained("unc-nlp/lxmert-base-uncased")
    
    @staticmethod
    def _load_image_features(root, image_id):
        """
        Loads the bounding boxes and features extracted from Faster RCNN for LXMERT
        
        :param root: path to feature
        :param image_id: image id in COCO
        :return: Tuple[np.ndarray, np.ndarray] normalized bounding boxes and ROI pooled features
        """
        def __reform_faster_features(saved):
            """
            Unpacks data from loaded .npz file and prepares it for LXMERT format
            
            :param saved: loaded .npz file
            :return: Tuple[np.ndarray, np.ndarray] normalized bounding boxes and ROI pooled features
            """
            
            # Unpack loaded numpy object
            height = saved['image_h']
            width = saved['image_w']
            boxes = saved['boxes']
            features = saved['features']
            
            # Calculate the components
            x = (boxes[:, 0:1] + boxes[:, 2:3]) / 2
            y = (boxes[:, 1:2] + boxes[:, 3:4]) / 2
            w = (boxes[:, 2:3] - boxes[:, 0:1])
            h = (boxes[:, 3:4] - boxes[:, 1:2])
            
            # Normalize and concatenate
            reformed_boxes = np.concatenate((x/width, y/height, w/width, h/height), axis=1)
            
            return features, reformed_boxes
            
        # Load cached features
        faster_features = np.load(os.path.join(root, f'{image_id}.npz'))
        return __reform_faster_features(faster_features)
    
    def _load_unaligned_features(self, true_img_id):
        """
        Randomly loads a different image's features than that of true_image_id
        
        :param root: 
        :param true_image_id: image id in COCO to avoid loading of
        :return: Tuple[np.ndarray, np.ndarray] Faster RCNN features of image that is not true_image_id
        """
        
        # All of the possible image_ids to sample
        candidate_img_ids1 = list({anns["image_id"] for anns in self.coco[0].anns.values() if anns["image_id"] != true_img_id})
        bp = len(candidate_img_ids1)
        
        # If 'restval' is being used or not
        if len(self.coco) == 2:
            candidate_img_ids2 = list({anns["image_id"] for anns in self.coco[1].anns.values() if anns["image_id"] != true_img_id})
            candidate_img_ids = candidate_img_ids1 + candidate_img_ids2
        else:
            candidate_img_ids = candidate_img_ids1
            
        # Sample a random index
        idx = random.choice(range(len(candidate_img_ids)))
        root = self.root[0] if idx < bp else self.root[1]
        return self._load_image_features(root, candidate_img_ids[idx])
        
    
    def __getitem__(self, index):
        """
        Gets the caption, Faster RCNN features, and whether the caption and image are aligned for index
        
        :param index: index in the list of caption_ids
        :return: Tuple[str, numpy.ndarray, numpy.ndarray, int]
        """
        
        # If the index is less than the breakpoint, it is part of the 
        # original training split.
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
            
        # Annotation id to be retrieved
        ann_id = self.ids[index]
        
        # Load corresponding COCO data
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        visual_feats, visual_pos = self._load_image_features(root, img_id)
        aligned = 1.0
        
        # Randomly unalign pairing with probability of prob_unaligned
        if random.random() < self.prob_unaligned:
            visual_feats, visual_pos = self._load_unaligned_features(img_id)
            aligned = 0.0
            
        outputs = self.tokenizer(caption, truncation=True, padding=True)
        outputs = {key: torch.tensor(value) for key, value in outputs.items()}
        outputs['visual_feats'] = torch.tensor(visual_feats)
        outputs['visual_pos'] = torch.tensor(visual_pos)
        
        if self.testing:
            outputs['index'] = torch.tensor(index)
        else:
            outputs['labels'] = torch.tensor(aligned)
            
        return outputs
    
    def __len__(self):
        """
        Returns length of dataset
        
        :return: int length of dataset
        """
        return len(self.ids)
