import argparse
import logging
import os
import sys
import shutil

from datasets.coco import get_test_dataset, get_train_dataset
from models.lxmert import LxmertForTBIR
from validate import i2t, t2i, encode_data

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange, tqdm

from transformers import set_seed
from transformers.models.lxmert import LxmertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import get_last_checkpoint, is_main_process


def rank_captions(model, all_input_ids, all_token_type_ids, 
                  all_attention_masks, query_visual_feats, query_visual_pos):
    model.eval()
    
    ranks = torch.zeros((100, all_input_ids.size(0))).cuda()
    
    for i, (visual_feats, visual_pos) in enumerate(zip(query_visual_feats, query_visual_pos)):
        scores = torch.zeros(all_input_ids.size(0)).cuda()
        
        for j in range(0, all_input_ids.size(0), 100):
            with torch.no_grad():
                batch_scores = model(all_input_ids[j:j+100],
                                     visual_feats.expand(100, *visual_feats.shape),
                                     visual_pos.expand(100, *visual_pos.shape),
                                     all_attention_masks[j:j+100],
                                     all_token_type_ids[j:j+100])
                scores[j:j+100] = batch_scores
                
        ranks[i] = scores
    
    model.train()
    
    return ranks


def rank_images(model, query_input_ids, query_token_type_ids, query_attention_masks,
                all_visual_feats, all_visual_pos):
    model.eval()
    
    ranks = torch.zeros((100, all_visual_feats.size(0))).cuda()
    
    for i, (input_ids, token_type_ids, attention_mask) in enumerate(zip(query_input_ids, query_token_type_ids, query_attention_masks)):
        scores = torch.zeros(all_visual_feats.size(0)).cuda()
        
        for j in range(0, all_visual_feats.size(0), 100):
            with torch.no_grad():
                batch_scores = model(input_ids.expand(100, *input_ids.shape),
                                     all_visual_feats[j:j+100],
                                     all_visual_pos[j:j+100],
                                     attention_mask.expand(100, *attention_mask.shape),
                                     token_type_ids.expand(100, *token_type_ids.shape))
                scores[j:j+100] = batch_scores
                
        ranks[i] = scores

    model.train()
    
    return ranks


def validate(val_loader, tokenizer, model):
    # compute the encoding for all the validation images and captions
    all_input_ids, all_token_type_ids, all_attention_masks, all_visual_feats, all_visual_pos = encode_data(tokenizer, 
                                                                                                           val_loader, 
                                                                                                           logging=logger.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(model, all_input_ids, all_token_type_ids, all_attention_masks, all_visual_feats, all_visual_pos)
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(model, all_input_ids, all_token_type_ids, all_attention_masks, all_visual_feats, all_visual_pos)
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
    parser.add_argument('--use_restval',
                        default=None,
                        type=bool,
                        required=True,
                        help='Whether to use the restval split from Karpathy et al. in training')
    parser.add_argument('--prob_unaligned',
                        default=None,
                        type=float,
                        required=True,
                        help='Probability that the images for each caption are randomly sampled from the negative images.')
    parser.add_argument("--do_train",
                        action='store_true',
                        required=True,
                        help="Whether to run training.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    
    ## Optional parameters
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
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
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
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
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    
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
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    set_seed(args.seed)
    
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and (n_gpu > 1 and torch.distributed.get_rank() == 0  or n_gpu <= 1):
        os.makedirs(args.output_dir)

    tokenizer = LxmertTokenizerFast.from_pretrained('unc-nlp/lxmert-base-uncased')

    #train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        print(f"Loading COCO Train Dataset {'with' if args.use_restval else 'without'} restval")
        train_dataset = get_train_dataset(args)
        val_dataset = get_test_dataset('val', args)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = LxmertForTBIR.from_pretrained('unc-nlp/lxmert-base-uncased', num_labels=1)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = SequentialSampler(val_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, batch_size=args.train_batch_size)
        val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, batch_size=args.train_batch_size)

        model.train()
        
        best_rsum = 0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch[0] = tokenizer(batch[0], padding=True, return_tensors='pt')
                batch = tuple(t.to(device) for t in batch)
                tokenized, visual_feats, visual_pos, labels = batch
                outputs = model(visual_feats=visual_feats, visual_pos=visual_pos, labels=labels, **tokenized)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += tokenized['input_ids'].size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                # evaluate on validation set
                rsum = validate(val_dataloader, tokenizer, model)

                # remember best R@ sum and save checkpoint
                is_best = rsum > best_rsum
                best_rsum = max(rsum, best_rsum)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'best_rsum': best_rsum
                }, is_best, prefix=args.output_dir + '/')

        # Save a trained model
        if args.do_train and (n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <=1):
            logger.info("******* Saving fine - tuned model *******")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
