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
    
    
def compute_ranks(labels: np.ndarray, logits: np.ndarray, num_captions_per_img: int, output_file: Optional[str] = None) -> Tuple[List[int], List[int]]:
    """Computes the rankings between every pairing of images and text and the rank of the first ground truth
    ground-truth pair.

    Args:
        labels (np.ndarray): The labels of whether a given pair is a ground-truth pair.
        logits (np.ndarray): The logits produced by the model when computing the matching score for a given pair
        num_captions_per_img (int): The number of captions per image (potentially different for 1k test split vs. 5k test split).
        output_file (str, optional): The output file to dump the ranks and matching matrices. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The ranks of ground-truth pairs for text retrieval and image retrieval respectively.
    """
    
    # Each row is an image and each column is a caption
    labels = labels.reshape(-1, num_captions_per_img)
    logits = logits.reshape(-1, num_captions_per_img)
    
    # Sort the logits along each column in descending order, so an image is the query
    inds = (-logits).argsort(axis=1)
    # Sort the labels along each row given by the rank predictions
    sorted_labels = np.take_along_axis(labels, inds, axis=1)
    # Get the first ground-truth pair across all of the captions
    i2t_ranks = (sorted_labels == 1).argmax(axis=1)
    
    # Sort the logits along each column in descending order, so a caption is the query
    inds = (-logits).argsort(axis=0)
    # Sort the labels along each column given by the rank predictions
    sorted_labels = np.take_along_axis(labels, inds, axis=0)
    # Get the first ground-truth pair across all of the images
    t2i_ranks = (sorted_labels == 1).argmax(axis=0)
    
    # Dump to the output_file?
    if output_file is not None:
        np.savez(output_file, i2t_ranks=i2t_ranks, t2i_ranks=t2i_ranks, labels=labels, logits=logits)

    return i2t_ranks, t2i_ranks

    
def compute_metrics_maker(num_captions_per_img: int, output_file: Optional[str] = None) -> Callable[[EvalPrediction], Dict]:
    """Creates a closure that takes number of captions per image and the output file and creates the metrics based off of
    the HuggingFace Trainer API.

    Args:
        num_captions_per_img (int): The number of captions per image (potentially different for 1k test split vs. 5k test split).
        output_file (str, optional): The output file to dump the ranks and matching matrices. Defaults to None.

    Returns:
        Callable[[EvalPrediction], Dict]: Function that calculates metrics as defined in the HuggingFace Trainer API.
    """
    
    def _compute_metrics(predictions: EvalPrediction) -> Dict:
        """Computes the metrics with the specified number of captions per image and to dump the rank matrics to the specified output file.

        Args:
            predictions (EvalPrediction): Predictions that the Trainer API provides.

        Returns:
            Dict: Dictionary of the different calculated metrics
        """
        
        # Get the ranks of the first ground-truth
        i2t_ranks, t2i_ranks = compute_ranks(predictions.label_ids, predictions.predictions, num_captions_per_img, output_file)
        
        # Care about recall@1, recall@5, and recall@10
        rank = [1, 5, 10]
        
        # The proportion of images where the first retrieved ground-truth caption is retrieved within the top k
        i2t_accs = [sum([r < k for r in i2t_ranks]) / len(i2t_ranks) for k in rank]
        eval_result = {"i2t_R@1": i2t_accs[0], "i2t_R@5": i2t_accs[1], "i2t_R@10": i2t_accs[2]}
        
        # The proportion of images where the retrieved ground-truth image is retrieved within the top k
        t2i_accs = [sum([r < k for r in t2i_ranks]) / len(t2i_ranks) for k in rank]
        eval_result["t2i_R@1"] = t2i_accs[0]
        eval_result["t2i_R@5"] = t2i_accs[1]
        eval_result["t2i_R@10"] = t2i_accs[2]
        
        # An easier proxy to make sure validation scores are increasing.
        eval_result["rsum"] = sum(eval_result.values())
        
        return eval_result
    
    return _compute_metrics
