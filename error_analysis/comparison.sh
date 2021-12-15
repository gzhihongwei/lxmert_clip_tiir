#!/bin/bash
#
#SBATCH --job-name=tiir-comparison
#SBATCH --output=%j.log
#SBATCH -e %j.err
#SBATCH --partition=m40-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=47GB
#SBATCH --exclude=node054,node059,node083,node084,node094,node095,node105,node108

. ../venv/bin/activate

# ============= Arguments  =========== #
DATA_PATH=$WORK_BASE/datasets/coco_ir/

python3 comparison.py \
    --data_path $DATA_PATH \
    --cross_image_eval \
    --eval_img_keys_file test_img_keys_1k.tsv