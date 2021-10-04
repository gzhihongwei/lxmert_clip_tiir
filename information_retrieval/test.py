import argparse
import logging
import os
import sys

from datasets.coco import get_test_dataset
from models.lxmert import LxmertForTBIR

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import set_seed
from transformers.models.lxmert import LxmertTokenizerFast
from transformers.trainer_utils import is_main_process

from train import validate
      
def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument('--data_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to the data directory that has coco')
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    
    ## Optional parameters
    parser.add_argument("--test_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    args.use_restval = False
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    set_seed(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # if not os.path.exists(args.output_dir) and (n_gpu > 1 and torch.distributed.get_rank() == 0  or n_gpu <= 1):
    #     os.makedirs(args.output_dir)

    tokenizer = LxmertTokenizerFast.from_pretrained('unc-nlp/lxmert-base-uncased')

    test_dataset = get_test_dataset('test', args)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'model_best.pth.tar'))
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Prepare model
    model = LxmertForTBIR.from_pretrained('unc-nlp/lxmert-base-uncased', num_labels=1, state_dict=checkpoint['model']).to(device)
    
    # torch.cuda.manual_seed_all(args.seed)
    # model = DDP(model, device_ids=[rank])
        
   
    # test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size)
    
    validate(test_dataloader, tokenizer, model, logger)


if __name__ == "__main__":
    main()
