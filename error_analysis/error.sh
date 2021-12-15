#!/bin/bash
#
#SBATCH --job-name=tiir-error
#SBATCH --output=%j.log
#SBATCH -e %j.err
#SBATCH --partition=titanx-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=47GB
#SBATCH --exclude=node054,node059,node083,node084,node094,node095,node105,node108

. ../venv/bin/activate

# ============= Arguments  =========== #
DATA_PATH=$WORK_BASE/datasets/coco_ir/

python3 error_analysis.py \
    --data_path $DATA_PATH \
    --cross_image_eval \
    --evaluation_output_file lxmert_1k_test.npz \
    --eval_img_keys_file test_img_keys_1k.tsv \
    --img_idxs 119 754 \
    --caption_idxs 2647 \
    --output_file lxmert_retrievals_for_clip.json
