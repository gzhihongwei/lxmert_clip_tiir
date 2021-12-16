#!/bin/bash
#
#SBATCH --job-name=contrastive
#SBATCH --output=%j.log
#SBATCH -e %j.err
#SBATCH --partition=2080ti-long
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
PROB_UNALIGNED=0
MARGIN=0.7
BATCH_SIZE=64
LEARNING_RATE=1e-5
NUM_EPOCHS=10
WEIGHT_DECAY=1e-4
N_GPU=8
METRIC=rsum

# Run 8 training processes on the 8 separate GPUs allocated
python3 -m torch.distributed.run --nproc_per_node=$N_GPU train.py \
            --model_name_or_path unc-nlp/lxmert-base-uncased \
            --formulation $FORMULATION \
            --use_fast_tokenizer \
            --margin $MARGIN \
            --top_k_violations $TOP_K \
            --data_path $DATA_PATH \
            --prob_unaligned $PROB_UNALIGNED \
            --cross_image_eval \
            --do_eval \
            --evaluation_strategy epoch \
            --per_device_eval_batch_size $BATCH_SIZE \
            --log_level info \
            --log_level_replica warning \
            --fp16 \
            --dataloader_num_workers 3 \
            --output_dir $OUTPUT_DIR \
            --overwrite_output_dir \
            --evaluate_during_training \
            --do_train \
            --per_device_train_batch_size $BATCH_SIZE \
            --save_strategy epoch \
            --weight_decay $WEIGHT_DECAY \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs $NUM_EPOCHS \
            --load_best_model_at_end \
            --metric_for_best_model $METRIC
