import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import LxmertModel, LxmertPreTrainedModel
from transformers.file_utils import ModelOutput


@dataclass
class LxmertForTBIROutput(ModelOutput):
    """
    Output type of :class:`LxmertForTBIR`.

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
    

class LxmertTBIRHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_labels),
        )
        
    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class LxmertForTBIR(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        
        # Lxmert backbone
        self.lxmert = LxmertModel(config)
        
        self.match_head = LxmertTBIRHead(config, self.config.num_labels)
        
        # Weight initialization
        self.init_weights()
        
        # Loss function
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        input_ids=None,
        visual_feats=None,
        visual_pos=None,
        attention_mask=None,
        token_type_ids=None,
        visual_attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        pooled_output = lxmert_output[2]
        matching_score = self.match_head(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss(matching_score, labels)

        if not return_dict:
            output = (matching_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        return LxmertForTBIROutput(
            loss=loss,
            matching_score=matching_score, 
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )
        