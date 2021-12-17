"""
Makes a copy of all images in the split specified renamed to the image ID assigned by MSCOCO.
"""

import argparse
import shutil

from pathlib import Path

import numpy as np

from pycocotools.coco import COCO
from tqdm import tqdm


def rename_coco(args: argparse.Namespace) -> None:
    """Renames a copy of theMSCOCO images in a split to their image ID.

    Args:
        args (argparse.Namespace): Command line arguments specifying the split, image directory, and output_directory.
    """
    
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
    renamed_dir = Path('datasets') / 'coco' / 'renamed' / args.split
    
    # Overri
    if args.output_dir is not None:
        renamed_dir = args.output_dir / args.split
        
    renamed_dir.mkdir(parents=True, exist_ok=True)
    
    print('Created directories to dump to')
    
    # Make a copy of the image renamed to their image id
    for image_filename, image_id in tqdm(zip(image_filenames, image_ids)):
        shutil.copy(args.image_dir / args.split / image_filename, renamed_dir / f"{image_id}.jpg")
        
    print('Done moving all files')
        

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True, help="COCO image root directory.")
    parser.add_argument('--split', type=str, required=True, help="Which split to rename to the COCO image ids.")
    parser.add_argument('--output_dir', type=Path, required=False, default=None, help="Where to output renamed images to.")
    args = parser.parse_args()

    rename_coco(args)
