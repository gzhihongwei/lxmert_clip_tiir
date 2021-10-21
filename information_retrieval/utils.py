from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from transformers.file_utils import ModelOutput
from transformers.models.lxmert.configuration_lxmert import LxmertConfig
from transformers.trainer_utils import EvalPrediction


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
    margin: Optional[float] = field(
        default=0.2,
        metadata={"help": "Margin used in the contrastive loss."}
    )
    max_violation: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use the maximum in batch negative violation as the loss."}
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
    n_gpu: int = field(
        metadata={"help": "Number of GPUs being used for training, evaluation, or prediction."}
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
    
    
class LxmertForIRConfig(LxmertConfig):
    def __init__(self, margin=0.2, max_violation=True, **kwargs):
        self.margin = margin
        self.max_violation = max_violation
        super().__init__(**kwargs)


@dataclass
class LxmertForIROutput(ModelOutput):
    """
    Output type of :class:`LxmertForIR*`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.k.
        matching_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`, `optional`):
            Scores of image-text matching objective (binary classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    matching_score: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    
def compute_ranks(labels: np.ndarray, logits: np.ndarray, num_captions_per_img: int) -> Tuple[List[int], List[int]]:
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
    return i2t_ranks, t2i_ranks

    
def compute_metrics_maker(num_captions_per_img: int) -> Callable[[EvalPrediction], Dict]:
    def _compute_metrics(predictions: EvalPrediction) -> Dict:
        i2t_ranks, t2i_ranks = compute_ranks(predictions.label_ids, predictions.predictions, num_captions_per_img)
        
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