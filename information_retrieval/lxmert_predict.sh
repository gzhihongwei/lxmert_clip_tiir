#!/bin/bash
#
#SBATCH --job-name=LXMERT-BCE-test-short
#SBATCH -o %j.log
#SBATCH -e %j.err
#SBATCH --partition=2080ti-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --mem=376GB

# Activate appropriate environment
. ../venv/bin/activate

# ============= Arguments  =========== #
DATA_PATH=$WORK_BASE/datasets/coco_ir/ 
FORMULATION=binary
OUTPUT_DIR=runs/lxmert/$FORMULATION
LOAD_PATH=runs/lxmert/$FORMULATION/checkpoint-10115
BATCH_SIZE=256
N_GPU=8

# Run prediction on 8 processes on the 8 separate GPUs allocated
python3 -m torch.distributed.run --nproc_per_node=$N_GPU lxmert/train.py \
            --model_name_or_path $LOAD_PATH \
            --formulation $FORMULATION \
            --use_fast_tokenizer \
            --data_path $DATA_PATH \
            --prob_unaligned 0 \
            --cross_image_eval \
            --per_device_eval_batch_size $BATCH_SIZE \
            --log_level info \
            --log_level_replica warning \
            --fp16 \
            --dataloader_num_workers 3 \
            --output_dir $OUTPUT_DIR \
            --overwrite_output_dir \
            --do_pred \
            --evaluation_output_file lxmert_1k_test.npz \
            --eval_img_keys_file test_img_keys_1k.tsv
            #--do_eval \
            #--evaluate_during_training
            #--margin $MARGIN \
            #--max_violation \
