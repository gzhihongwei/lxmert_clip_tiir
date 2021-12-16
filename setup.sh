#!/bin/bash

# Get COCO dataset images
wget -P datasets/coco/ http://images.cocodataset.org/zips/train2014.zip
unzip datasets/coco/train2014.zip -d datasets/coco/images
rm datasets/coco/train2014.zip

wget -P datasets/coco http://images.cocodataset.org/zips/val2014.zip
unzip datasets/coco/val2014.zip -d datasets/coco/images
rm datasets/coco/val2014.zip

# Get COCO dataset annotations (if preprocessing with Faster R-CNN)
wget -P datasets/coco http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip datasets/coco/annotations_trainval2014.zip -d datasets/coco/
rm datasets/coco/annotations_trainval2014.zip

# Download the captions, image keys, etc.
azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/coco_ir' datasets --recursive
rm -rf datasets/coco_ir/*.json datasets/coco_ir/*labels.pt datasets/coco_ir/gt_objstuff datasets/coco_ir/panderson_nmfilter2_thres0.2_imsperbatch1_tsv

# Create a virtual environment in the repo root for handling the repos dependencies
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CUDA 11.1 is assumed)
pip3 install -r requirements.txt

# Stringify the keys in coco_ir
python3 preprocess/stringify_keys.py datasets/coco_ir

# Install this repo as a package
pip3 install .

# Preprocess images for CLIP
python3 preprocess/coco_image_rename.py --image_dir datasets/coco/images \
  --split val2014 \
  --output_dir datasets/coco/renamed

python3 preprocess/image_to_h5.py --captions_dir datasets/coco_ir \
  --val2014_loc datasets/coco/renamed/val2014 \
  --output_dir datasets/coco_ir/ \
  --splits minival val test

# Done preprocessing for CLIP
deactivate

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#             OPTIONAL: COMMENT IN IF YOU WANT TO PREPROCESS THE IMAGES FOR LXMERT          #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# cd faster_rcnn

# # Create a separate environment for the submodule
# python3 -m venv venv
# source venv/bin/activate

# # Get the pretrained model and put it somewhere
# pip3 install gdown
# gdown https://drive.google.com/uc?id=18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN -O models/

# # Install dependencies
# pip3 install -r requirements.txt

# # Compile CUDA extensions
# cd lib
# python setup.py build develop
# cd ../../

# # Preprocess images through Faster R-CNN for LXMERT
# python3 preprocess/get_coco_features.py --dataset vg \
#   --net res101 \
#   --cfg faster_rcnn/cfgs/res101.yml \
#   --classes_dir faster_rcnn/data/genome/1600-400-20 \
#   --load_dir faster_rcnn/models \
#   --cuda \
#   --image_dir datasets/coco/images \
#   --output_dir datasets/coco

# # Move the Faster features to an .h5 file
# python3 preprocess/faster_to_h5.py --image_dir datasets/coco/images/faster_features \
#   --captions_dir datasets/coco_ir \
#   --output_dir datasets/coco_ir

# # Done preprocessing
# deactivate

# Move all split related stuff to their respective folders
for split in "train" "minival" "val" "test"; do
  mkdir -p datasets/coco_ir/$split
  mv datasets/coco_ir/"${split}_*" datasets/coco_ir/$split
done
