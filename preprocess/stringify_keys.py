"""
Stringify the keys provided by Microsoft Oscar.
"""

import argparse

from pathlib import Path

import torch


if __name__ == "__main__":
    # Required argument of the captions directory
    parser = argparse.ArgumentParser()
    parser.add_argument("captions_dir", type=Path, help="The directory where the caption files from Microsoft Oscar are.")
    args = parser.parse_args()
    
    # Find all of the caption files
    for caption_file in args.captions_dir.glob("*_captions.pt"):
        print(caption_file)
        captions = torch.load(caption_file)
        
        # Stringify their keys
        for key in captions:
            captions[str(key)] = captions.pop(key)
            
        torch.save(captions, caption_file)
