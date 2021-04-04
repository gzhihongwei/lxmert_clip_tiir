""" Usage: srun -p 1080ti-short --gres=gpu:1 python3 get_coco_features.py --dataset vg --net res101 --cfg ../../deep_tbir/models/faster_rcnn/cfgs/res101.yml --classes_dir ../../deep_tbir/models/faster_rcnn/data/genome/1600-400-20 --load_dir ../../deep_tbir/models/faster_rcnn/models --cuda
"""

import os
import numpy as np

from pycocotools.coco import COCO
from generate_features import parse_args, load_model, get_detections_from_im
from tqdm import tqdm


def featurize_coco(split, img_root):
    # Make sure the split is one in COCO 2014
    assert split in {'train2014', 'val2014', 'test2014'}, 'split must be either "train2014", "val2014", or "test2014"'
    
    # Get the associated annotation file
    if split == 'train2014':
        ann_file = 'annotations/captions_train2014.json'
    elif split == 'val2014':
        ann_file = 'annotations/captions_val2014.json'
    elif split == 'test2014':
        ann_file = 'annotations/image_info_test2014.json'
        
    # Create COCO API for given annotation file
    coco = COCO(ann_file)
    
    print('Loaded COCO annotation file')
    
    # Get image ids and filenames
    image_ids = coco.getImgIds()
    image_filenames = [image['file_name'] for image in coco.loadImgs(image_ids)]
    
    # Create the place to store the features extracted from Faster-RCNN
    faster_dir = os.path.join('faster_features', split)
    os.makedirs(faster_dir, exist_ok=True)
    
    print('Created directories to dump to')
    
    # Load the pre-trained model
    faster_args = parse_args()
    classes, faster_rcnn = load_model(faster_args)
    
    # Featurize each image in the COCO split
    for image_filename, image_id in tqdm(zip(image_filenames, image_ids)):
        detections = get_detections_from_im(faster_rcnn, classes, os.path.join(img_root, image_filename), image_id, faster_args)
        np.savez_compressed(os.path.join(faster_dir, f'{image_id}.npz'), **detections)
        
    print('Done obtaining all Faster-RCNN detections')
        

if __name__ == '__main__':    
    os.chdir(os.path.join("..", "..", "datasets", "coco"))
    split = 'train2014'
    img_root = os.path.join("images", split)
    featurize_coco(split, img_root)
        
        
        
    
    
    
    
