#!/bin/bash
#
#SBATCH --job-name=BCE-test
#SBATCH --output=%j.log
#SBATCH -e %j.err
#SBATCH --partition=1080ti-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --mem=376GB
#SBATCH --exclude=node051,node059,node029,node060
#2080ti-long, gres=gpu:8, 376GB

TEST_SPLIT=test
mkdir -p /local/gzwei/datasets/coco_ir

rsync --ignore-existing -av $WORK_BASE/datasets/coco_ir/$TEST_SPLIT /local/gzwei/datasets/coco_ir

. ../venv/bin/activate

# ============= Arguments  =========== #
DATA_PATH=/local/gzwei/datasets/coco_ir/ 
FORMULATION=binary
OUTPUT_DIR=runs/$FORMULATION
LOAD_PATH=runs/$FORMULATION/checkpoint-10115
#runs/contrastive/checkpoint-11070 #runs/binary/checkpoint-10115
#MARGIN=0.3
BATCH_SIZE=256
N_GPU=8

python3 -m torch.distributed.run --nproc_per_node=$N_GPU train.py \
            --model_name_or_path $LOAD_PATH \
            --formulation $FORMULATION \
            --use_fast_tokenizer \
            --data_path $DATA_PATH \
            --cross_image_eval \
            --per_device_eval_batch_size $BATCH_SIZE \
            --log_level info \
            --log_level_replica warning \
            --fp16 \
            --dataloader_num_workers 3 \
            --output_dir $OUTPUT_DIR \
            --overwrite_output_dir \
            --do_pred
            #--eval_img_keys_file test_img_keys_1k.tsv
            #--do_eval \
            #--evaluate_during_training
            #--margin $MARGIN \
            #--max_violation \
