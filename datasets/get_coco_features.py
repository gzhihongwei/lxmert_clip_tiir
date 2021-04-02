import argparse
import os
import numpy as np

from pycocotools.coco import COCO
from generate_features import parse_args, load_model, get_detections_from_im
from tqdm import tqdm


def featurize_coco(args):
    # Make sure the split is one in COCO 2014
    assert args.split in {'train2014', 'val2014', 'test2014'}, 'split must be either "train2014", "val2014", or "test2014"'
    
    # Get the associated annotation file
    if args.split == 'train2014':
        ann_file = 'annotations/captions_train2014.json'
    elif args.split == 'val2014':
        ann_file = 'annotations/captions_val2014.json'
    elif args.split == 'test2014':
        ann_file = 'annotations/image_info_test2014.json'
        
    # Create COCO API for given annotation file
    coco = COCO(ann_file)
    
    print('Loaded COCO annotation file')
    
    # Get image ids and filenames
    image_ids = coco.getImgIds()
    image_filenames = [image['file_name'] for image in coco.loadImgs(image_ids)]
    
    # Create the place to store the features extracted from Faster-RCNN
    faster_dir = os.path.join('faster_features', args.split)
    os.makedirs(faster_dir, exist_ok=True)
    
    print('Created directories to dump to')
    
    # Load the pre-trained model
    faster_args = parse_args()
    classes, faster_rcnn = load_model(faster_args)
    
    # Featurize each image in the COCO split
    for image_filename, image_id in tqdm(zip(image_filenames, image_ids)):
        detections = get_detections_from_im(faster_rcnn, classes, os.path.join(args.img_root, image_filename), image_id, faster_args)
        np.save(os.path.join(faster_dir, f'{image_id}.npz'), **detections)
        
    print('Done obtaining all Faster-RCNN detections')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obtains Faster-RCNN pooled features and bounding boxes for COCO 2014 splits')
    parser.add_argument('-c', '--chdir', type=str, 
                      dest='change_dir', default=os.path.join("..", "..", "datasets", "coco"),
                      help='Path to change to before featurizing')
    parser.add_argument('-s', '--split', type=str, 
                      dest='split', required=True,
                      help='Which split to featurize the images of')
    parser.add_argument('-i', '--image_root', type=str,
                      dest='image_root', required=True,
                      help='Relative path from change_dir')
    
    args = parser.parse_args()
    
    os.chdir(args.change_dir)
    featurize_coco(args)
        
        
        
    
    
    
    
