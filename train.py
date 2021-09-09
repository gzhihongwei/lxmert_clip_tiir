import argparse
import logging
import os
import sys
import shutil

# import torch_distrib

from datasets.coco import get_test_dataset, get_train_dataset
from models.lxmert import LxmertForTBIR
from validate import i2t, t2i, encode_data

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange, tqdm

from transformers import set_seed
from transformers.models.lxmert import LxmertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import get_last_checkpoint, is_main_process


def validate(val_loader, tokenizer, model, logger):
    # compute the encoding for all the validation images and captions
    all_input_ids, all_token_type_ids, all_attention_masks, all_visual_feats, all_visual_pos = encode_data(tokenizer, 
                                                                                                           val_loader, 
                                                                                                           logging=logger.info)

    logger.info('Starting image to text')
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(model, all_input_ids, all_token_type_ids, all_attention_masks, all_visual_feats, all_visual_pos, logging=logger.info)
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    
    logger.info('Starting text to image')
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(model, all_input_ids, all_token_type_ids, all_attention_masks, all_visual_feats, all_visual_pos, logging=logger.info)
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        

def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument('--data_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to the data directory that has coco')
    parser.add_argument('--prob_unaligned',
                        default=None,
                        type=float,
                        required=True,
                        help='Probability that the images for each caption are randomly sampled from the negative images.')
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    
    ## Optional parameters
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='Local rank of a process on a node.')
    parser.add_argument('--use_restval',
                        action='store_true',
                        help='Whether to use the restval split from Karpathy et al. in training')
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Batch size per process.")
    parser.add_argument("--learning_rate",
                        default=3e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", 
                        default=0, 
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    
    args = parser.parse_args()
    print(args)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if is_main_process(args.local_rank) else logging.WARN)
    
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=dist.get_world_size(),
                            rank=args.local_rank)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    set_seed(args.seed)
    
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and (n_gpu > 1 and dist.get_rank() == 0  or n_gpu <= 1):
        os.makedirs(args.output_dir)

    tokenizer = LxmertTokenizerFast.from_pretrained('unc-nlp/lxmert-base-uncased')

    logger.info(f"Loading COCO Train Dataset {'with' if args.use_restval else 'without'} restval")
    train_dataset = get_train_dataset(args)
    val_dataset = get_test_dataset('val', args)
    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if n_gpu > 1:
        num_train_optimization_steps = num_train_optimization_steps // dist.get_world_size()

    # Prepare model
    model = LxmertForTBIR.from_pretrained('unc-nlp/lxmert-base-uncased', num_labels=1)
    if args.fp16:
        scaler = GradScaler()
    model.to(device)
    if n_gpu > 1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.batch_size * n_gpu)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    if n_gpu == 1:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        args.train_batch_size = args.batch_size
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=n_gpu, rank=args.rank)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=n_gpu, rank=args.rank)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=1, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    model.train()
    
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch[0] = tokenizer(list(batch[0]), padding=True, return_tensors='pt')
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            tokenized, visual_feats, visual_pos, labels = batch
            
            if args.fp16:
                with autocast():
                    outputs = model(visual_feats=visual_feats.float(), visual_pos=visual_pos.float(), labels=labels.float(), **tokenized)
                    loss = outputs[0]
            else:
                outputs = model(visual_feats=visual_feats.float(), visual_pos=visual_pos.float(), labels=labels.float(), **tokenized)
                loss = outputs[0]
                
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                
            # evaluate on validation set
            # rsum = validate(val_dataloader, tokenizer, model, logger)

        # remember best R@ sum and save checkpoint
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), f'{args.output_dir}/checkpoint-{epoch}.pth.tar')
            # is_best = rsum > best_rsum
            # best_rsum = max(rsum, best_rsum)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'model': model.state_dict(),
            #     'best_rsum': best_rsum
            # }, is_best, prefix=args.output_dir + '/')

    # Save a trained model
    if (n_gpu > 1 and dist.get_rank() == 0) or n_gpu == 1:
        logger.info("******* Saving fine - tuned model *******")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
