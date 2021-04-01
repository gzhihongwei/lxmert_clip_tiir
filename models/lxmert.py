import os
import sys

# Appended to find 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Faster-R-CNN-with-model-pretrained-on-Visual-Genome')))

import torch
import torchvision

from generate_tsv import parse_args, load_model
from transformers import LxmertTokenizer, LxmertForQuestionAnswering


# Need a tokenizer for input_ids
# 36 object ROIs from Faster-RCNN
# Need ROI pooled object features from bounding boxes from bounding boxes of Faster R-CNN
# Need visual positions normalized to 0 through 1
faster_rcnn = load_model(parse_args())
print(faster_rcnn)

# https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/blob/master/generate_tsv.py
# from generate_tsv import parse_args, load_model
