import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from transformers.models.lxmert.tokenization_lxmert_fast import LxmertTokenizerFast

from datasets.coco import get_test_dataset, get_train_dataset
from information_retrieval.lxmert import LxmertForIR
from information_retrieval.coco_ir import RetrievalDataset

import numpy as np
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    Trainer,
    TrainingArguments
)
from transformers.trainer_pt_utils import nested_concat, nested_numpify, nested_truncate, denumpify_detensorize
from transformers.trainer_utils import get_last_checkpoint, EvalLoopOutput


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    data_path: str = field(
        metadata={"help": "Path to the data directory that has COCO."}
    )
    prob_unaligned: float = field(
        metadata={"help": "Probability that the images for each caption are randomly sampled from the negative images."}
    )
    cross_image_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Perform cross image inference, i.e. each image with all texts from other images."}
    )
    eval_img_keys_file: Optional[str] = field(
        default='',
        metadata={"help": "Image key tsv to select a subset of images for evaluation. "
                          "This is useful in 5-folds evaluation. The topn index file is not " 
                          "needed in this case."}
    )
    eval_caption_index_file: Optional[str] = field(
        default='', 
        metadata={"help": "index of a list of (img_key, cap_idx) for each image."
                          "this is used to perform re-rank using hard negative samples."
                          "useful for validation set to monitor the performance during training."}
    )
    evaluate_during_training: Optional[bool] = field(
        default=False,
        metadata={"help": "Run evaluation during training at each save_steps."}
    )
    

class LxmertForIRTrainer(Trainer):
    def evaluation_loop(self, 
                        dataloader: torch.utils.data.dataloader.DataLoader, 
                        description: str, 
                        prediction_loss_only: Optional[bool] = None, 
                        ignore_keys: Optional[List[str]] = None, 
                        metric_key_prefix: str = 'eval') -> :
        self.model.eval()
        
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {len(dataloader)}")
        logger.info(f"  Batch size = {dataloader.batch_size}")
        
        results = {}
        all_labels = None
        for inputs in dataloader:
            indices = inputs.pop('index')
            labels = inputs['labels']
            labels = self._pad_across_processes(labels)
            labels = self._nested_gather(labels)
            labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            labels = nested_numpify(labels_host)
            all_labels = (
                labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
            )
            all_labels = nested_truncate(all_labels, len(dataloader))

            with torch.no_grad():
                _, results = self.model(**inputs)[:2]
                results = [result.to(torch.device("cpu")) for result in results]
                results.update({idx.item(): res.item() for idx, res in zip(indices, results)})
                
        metrics = LxmertForIRTrainer._evaluate(dataloader.dataset, results)
        predictions = (metrics.pop("i2t_ranks"), metrics.pop("t2i_ranks"))
        all_labels = denumpify_detensorize(all_labels)
        
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return EvalLoopOutput(predictions=predictions, label_ids=all_labels, metrics=metrics, num_samples=len(dataloader))
    
    @staticmethod
    def _evaluate(dataset: torch.utils.data.Dataset, test_results: Dict[int, float]) -> Dict[str, Dict[str, float]]:
        i2t_ranks, t2i_ranks = LxmertForIRTrainer._compute_ranks(dataset, test_results)
        rank = [1, 5, 10]
        i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
        eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
        eval_result["i2t_ranks"] = np.array(i2t_ranks)
        if t2i_ranks:
            t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
            eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
            eval_result["t2i_ranks"] = np.array(t2i_ranks)
        return eval_result
    
    @staticmethod
    def _compute_ranks(dataset: torch.utils.data.Dataset, results: Dict[int, float]) -> Tuple[List[int], Optional[List[int]]]:
        labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
        similarities = np.array([results[i] for i in range(len(dataset))])
        if dataset.has_caption_indexs:
            num_captions_per_img = dataset.num_captions_per_img
        else:
            num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
        labels = np.reshape(labels, (-1, num_captions_per_img))
        similarities = np.reshape(similarities, (-1, num_captions_per_img))
        i2t_ranks, t2i_ranks = [], []
        for lab, sim in zip(labels, similarities):
            inds = (-sim).argsort()
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            i2t_ranks.append(rank)
        if not dataset.has_caption_indexs:
            labels = np.swapaxes(labels, 0, 1)
            similarities = np.swapaxes(similarities, 0, 1)
            for lab, sim in zip(labels, similarities):
                inds = (-sim).argsort()
                rank = num_captions_per_img
                for r, ind in enumerate(inds):
                    if lab[ind] == 1:
                        rank = r
                        break
                t2i_ranks.append(rank)
        return i2t_ranks, t2i_ranks
            

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        num_labels=1
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = LxmertForIR.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    train_dataset = RetrievalDataset(tokenizer, data_args, 'train', is_train=True)
    val_dataset = RetrievalDataset(tokenizer, data_args, 'minival' if data_args.evaluate_during_training else 'val', is_train=False)
    test_dataset = RetrievalDataset(tokenizer, data_args, 'test', is_train=False)
    
    if not training_args.do_train and not (training_args.do_eval or training_args.do_predict):
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    
    # Initialize our Trainer
    trainer = LxmertForIRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else test_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
    if training_args.do_eval or training_args.do_predict:
        logger.info("*** Evaluate ***")
        split = "eval" if training_args.do_eval else "test"
        metrics = trainer.evaluate(metric_key_prefix=split)

        trainer.log_metrics(split, metrics)
        trainer.save_metrics(split, metrics)


if __name__ == "__main__":
    main()
