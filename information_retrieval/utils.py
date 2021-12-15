from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from transformers.trainer_utils import EvalPrediction


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    formulation: str = field(
        metadata={"help": "Which formulation to use for text-image retrieval. Must be one of {'binary', 'contrastive'}"}
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
    return_dict: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return dicts in model forward"}
    )
    ignore_keys: Optional[List[str]] = field(
        default=None,
        metadata={"help": "What keys to ignore during evaluation if return_dict=True"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    data_path: str = field(
        metadata={"help": "Path to the data directory that has COCO."}
    )
    prob_unaligned: Optional[float] = field(
        default=0,
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
    evaluate_during_training: Optional[bool] = field(
        default=False,
        metadata={"help": "Run evaluation during training at each save_steps."}
    )
    evaluation_output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Where to output the ranks in both directions, labels, and logits if specified"}
    )
    
    
def compute_ranks(labels: np.ndarray, logits: np.ndarray, num_captions_per_img: int, output_file: Optional[List[str]] = None) -> Tuple[List[int], List[int]]:
    labels = labels.reshape(-1, num_captions_per_img)
    logits = logits.reshape(-1, num_captions_per_img)
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, logits):
        inds = (-sim).argsort()
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    labels = labels.swapaxes(0, 1)
    logits = logits.swapaxes(0, 1)
    for lab, sim in zip(labels, logits):
        inds = (-sim).argsort()
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        t2i_ranks.append(rank)
    
    if output_file is not None:
        np.savez(output_file, i2t_ranks=i2t_ranks, t2i_ranks=t2i_ranks, labels=labels, logits=logits)

    return i2t_ranks, t2i_ranks, labels, logits

    
def compute_metrics_maker(num_captions_per_img: int, output_file: Optional[List[str]] = None) -> Callable[[EvalPrediction], Dict]:
    def _compute_metrics(predictions: EvalPrediction) -> Dict:
        i2t_ranks, t2i_ranks, labels, logits = compute_ranks(predictions.label_ids, predictions.predictions, num_captions_per_img, output_file)
        
        rank = [1, 5, 10]
        
        i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
        eval_result = {"i2t_R@1": i2t_accs[0], "i2t_R@5": i2t_accs[1], "i2t_R@10": i2t_accs[2]}
        
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        eval_result["t2i_R@1"] = t2i_accs[0]
        eval_result["t2i_R@5"] = t2i_accs[1]
        eval_result["t2i_R@10"] = t2i_accs[2]
        
        eval_result["rsum"] = sum(eval_result.values())
        
        return eval_result
    
    return _compute_metrics
