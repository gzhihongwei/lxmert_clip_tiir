from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from transformers.file_utils import ModelOutput
from transformers.models.lxmert.configuration_lxmert import LxmertConfig

from ..utils import ModelArguments


@dataclass
class LxmertModelArguments(ModelArguments):
    """
    Extra model arguments for possible extensions (i.e contrastive learning with varying the number of in batch negatives to consider
    for the loss).

    Args:
        margin (Optional[float]): The margin used in the contrastive loss.
        top_k_violations (Optional[int]): The number of the top k in batch negative violations as the loss.
    """
    
    margin: Optional[float] = field(
        default=0.2,
        metadata={"help": "Margin used in the contrastive loss. Define if using contrastive loss."}
    )
    top_k_violations: Optional[int] = field(
        default=None,
        metadata={"help": "Specify for the top k in batch negative violations as the loss."}
    )
    

@dataclass
class LxmertForIRConfig(LxmertConfig):
    def __init__(self, margin=0.2, top_k_violations=None, **kwargs):
        self.margin = margin
        self.top_k_violations = top_k_violations
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
