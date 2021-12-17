"""
Script that creates the .h5 image files for the minival, val, and test splits of COCO for CLIP.
"""
import argparse

from pathlib import Path

import h5py
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--captions_dir', type=Path, required=True, 
                        help="The root directory containing the pytorch caption files")
    parser.add_argument('-v', '--val2014_loc', type=Path, required=True, help="Where the renamed/val2014 directory is")
    parser.add_argument('-o', '--output_dir', type=Path, required=False, default=None,
                        help="Where to output preprocessed images to.")
    parser.add_argument('-s', '--splits', nargs='+', required=False, default=None,
                        help="Which splits to skip if applicable.")
    args = parser.parse_args()

    # All test/val images are in the val2014 split of COCO
    img_root = args.val2014_loc

    # Go through each split and create an .h5 file of all its images
    for split in args.splits:
        print("On split [{}]".format(split))
        captions = torch.load(args.captions_dir / f"{split}_captions.pt")
        img_keys = list(captions.keys())
        
        output_dir = args.captions_dir if args.output_dir is not None else args.output_dir
            
        output_dir /= split
        
        output_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_dir / f"{split}_imgs.h5", "w") as f:
            for img_key in tqdm(img_keys):
                img = np.asarray(Image.open(img_root / f"{img_key}.jpg"))
                f.create_dataset(str(img_key), data=img, dtype=np.float32)
