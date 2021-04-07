import numpy as np
import os
import random
import torch.utils.data as data


def get_loader(batch_size, feature_root, ann_file="annotations/captions_train2014.json", 
               prob_unaligned=0.9, num_workers=0):
    """
    Returns the pytorch dataset for either the training or validation split of COCO 2014 captions
    
    :param batch_size: batch size of dataloader
    :param feature_root: directory where all of the saved Faster RCNN features are located
    :param ann_file: annotation file for the data to be loaded
    :param prob_unaligned: probability that the Faster RCNN feature is not from the image
                           paired with the caption
    :param num_workers: number of other sub processes loading the data
    :return: torch.utils.data.DataLoader for the COCO dataset
    """
    
    # Gets the pytorch dataset
    dataset = CoCoText2Img(feature_root=feature_root, 
                           ann_file=ann_file, 
                           prob_unaligned=prob_unaligned)
    
    # Gets the corresponding pytorch dataloader
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    return data_loader


class CoCoText2Img(data.Dataset):
    """
    Pytorch dataset that handles COCO train2014 and val2014 splits
    """
    def __init__(self, feature_root, ann_file, prob_unaligned):
        """
        Constructor for CoCoText2Img
        
        :param feature_root: directory where all of the saved Faster RCNN features are located
        :param ann_file: annotation file for the data to be loaded
        :param prob_unaligned: probability that the Faster RCNN feature is not from the image
                               paired with the caption
        """
        from pycocotools.coco import COCO
        
        # Store the feature root
        self.feature_root = feature_root
        # Create a COCO instance
        self.coco = COCO(ann_file)
        # All of the caption ids
        self.ids = list(sorted(self.coco.anns.keys()))
        # Probability of returning an unaligned image from caption
        assert 0 <= prob_unaligned <= 1, "prob_unaligned must be a probability" 
        self.prob_unaligned = prob_unaligned
        
    def _get_caption_image_id(self, ann_id):
        """
        Gets the image_id of the image paired with caption of ann_id
        
        :param ann_id: caption id in COCO
        :return: int the image_id of the associated image to the caption
        """
        return self.coco.loadAnns(ann_id)[0]["image_id"]
    
    def _load_caption(self, ann_id):
        """
        Gets the caption with ann_id
        
        :param ann_id: caption id in COCO
        :return: str the caption
        """
        return self.coco.loadAnns(ann_id)[0]["caption"]
    
    def _load_image_features(self, image_id):
        """
        Loads the bounding boxes and features extracted from Faster RCNN for LXMERT
        
        :param image_id: image id in COCO
        :return: Tuple[np.ndarray, np.ndarray] normalized bounding boxes and ROI pooled features
        """
        def _reform_faster_features(saved):
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
            
            return reformed_boxes, features
            
        # Load cached features
        faster_features = np.load(os.path.join(self.feature_root, f'{image_id}.npz'))
        return _reform_faster_features(faster_features)
    
    def _load_unaligned_features(self, true_image_id):
        """
        Randomly loads a different image's features than that of true_image_id
        
        :param true_image_id: image id in COCO to avoid loading of
        :return: Tuple[np.ndarray, np.ndarray] Faster RCNN features of image that is not true_image_id
        """
        
        # All of the possible image_ids to sample
        candidate_image_ids = list({anns["image_id"] for anns in self.coco.anns.values() if anns["image_id"] != true_image_id})
        # Sample an image_id
        unaligned_image_id = random.choice(candidate_image_ids)
        return self._load_image_features(unaligned_image_id)
        
    
    def __getitem__(self, index):
        """
        Gets the caption, Faster RCNN features, and whether the caption and image are aligned for index
        
        :param index: index in the list of caption_ids
        :return: Tuple[str, ]
        """
        
        # Get the caption id at index of the list of caption ids
        id = self.ids[index]
        # Load corresponding COCO data
        caption = self._load_caption(id)
        image_id = self._get_caption_image_id(id)
        boxes, features = self._load_image_features(image_id)
        aligned = 1
        
        # Randomly unalign pairing with probability of prob_unaligned
        if random.random() < self.prob_unaligned:
            aligned = 0
            image = self._load_unaligned_features(id)
            
        return caption, image, aligned
    
    def __len__(self):
        """
        Returns length of dataset
        
        :return: int length of dataset
        """
        return len(self.ids)
        
        
if __name__ == "__main__":
    os.chdir(os.path.join("..", "..", "datasets", "coco"))
    train_loader = get_loader(batch_size=32,
                              root=os.path.join("images", "train2014"),
                              ann_file=os.path.join("annotations", "captions_train2014.json"))
    print(next(iter(train_loader)))
    
    
  
