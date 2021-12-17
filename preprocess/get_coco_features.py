"""
This file preprocesses each image in MSCOCO through the pretrained Faster R-CNN in the given submodule.
"""

import os
import sys
import numpy as np

from pycocotools.coco import COCO

# Change the path so that generate_features.py in the `faster_rcnn` submodule can be imported
sys.path.append(os.path.abspath("faster_rcnn"))

from generate_features import parse_args, load_model, get_detections_from_im
from tqdm import tqdm


def featurize_coco(args):
    # Make sure the split is one in COCO 2014
    assert args.split in {'train2014', 'val2014'}, 'split must be either "train2014" or "val2014"'
    
    # Get the associated annotation file
    if args.split == 'train2014':
        ann_file = 'captions_train2014.json'
    elif args.split == 'val2014':
        ann_file = 'captions_val2014.json'
        
    # Create COCO API for given annotation file
    coco = COCO(str(args.image_dir.parent / "annotations" / ann_file))
    
    print('Loaded COCO annotation file')
    
    # Get image ids and filenames
    image_ids = coco.getImgIds()
    image_filenames = [image['file_name'] for image in coco.loadImgs(image_ids)]
    
    # Create the place to store the features extracted from Faster-RCNN
    faster_dir = args.output_dir if args.output_dir is not None else args.image_dir.parent 
    faster_dir /= 'faster_features' / args.split
    faster_dir.mkdir(parents=True, exist_ok=True)
    
    print('Created directories to dump to')
    
    # Load the pre-trained model
    classes, faster_rcnn = load_model(args)
    
    # Featurize each image in the COCO split
    for image_filename, image_id in tqdm(zip(image_filenames, image_ids)):
        if (faster_dir / f'{image_id}.npz').isfile():
            continue
        detections = get_detections_from_im(faster_rcnn, classes, str(args.img_dir / args.split / image_filename), image_id, args, 0)
        np.savez_compressed(str(faster_dir / f'{image_id}.npz'), **detections)
        
    print('Done obtaining all Faster-RCNN detections')
        

if __name__ == '__main__':    
    args = parse_args()
    featurize_coco(args)
