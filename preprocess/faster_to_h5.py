"""
This file converts the dumped `.npz` Faster R-CNN features into `.h5` files to prevent loading all images into RAM at once.
"""

import argparse
import os

from pathlib import Path

import h5py
import numpy as np
import torch

from tqdm import tqdm


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=Path, required=True, 
                        help="The root directory containing the Faster R-CNN image features.")
    parser.add_argument('-c', '--captions_dir', type=Path, required=True, 
                        help="The root directory containing the PyTorch caption files from Microsoft Oscar.")
    parser.add_argument('-o', '--output_dir', type=Path, required=True,
                        help="Where to output preprocessed images to.")
    parser.add_argument('-s', '--skip_splits', nargs='+', required=False, default=None,
                        help="Which splits to skip if applicable.")
    args = parser.parse_args()

    # These caption files are from Microsoft Oscar (explained in README)
    for filepath in args.captions_dir.glob("*.pt"):
        filename = os.path.basename(filepath)
        split = filename.split("_")[0]
        
        # Skip split?
        if args.skip_splits is not None and split in args.skip_splits:
            continue

        print('='*60)
        print("On split [{}]".format(split))
        print('='*60 + '\n')
        
        captions_dict = torch.load(filepath)
        img_ids = list(captions_dict.keys())
        
        output_dir = args.output_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)

        # The `.h5` file for a given split
        with h5py.File(output_dir / f"{split}_img_frcnn_feats.h5", "w") as f:
            for img_id in tqdm(img_ids):
                # Creates a base group for each image id and then adds the visual_feats and visual_pos as datasets under it
                group = f.create_group(str(img_id))
                img_feats_path = list(args.image_dir.glob(f"**/{img_id}.npz"))[0]
                img_feats = np.load(img_feats_path)
                visual_feats = group.create_dataset("visual_feats", data=img_feats["visual_feats"], dtype=np.float32)
                visual_pos = group.create_dataset("visual_pos", data=img_feats["visual_pos"], dtype=np.float32)

            print("\n" + "=" * 60)
            print(f"Saving image features for [{split}]")
            print("=" * 60 + "\n")
            