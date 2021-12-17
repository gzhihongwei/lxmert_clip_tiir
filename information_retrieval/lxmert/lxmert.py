from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LxmertModel, LxmertPreTrainedModel
from transformers.models.lxmert.configuration_lxmert import LxmertConfig

from .utils import LxmertForIRConfig, LxmertForIROutput


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss as given in VSE++ (https://arxiv.org/abs/1707.05612), but allowing different top k violations.
    """

    def __init__(self, margin: int = 0, top_k_violations: Optional[int] = None):
        """
        Initialize contrastive loss.

        Args:
            margin (int, optional): The contrastive loss margin. Defaults to 0.
            top_k_violations (int, optional): The number of top k violations to consider when calculating the loss. Defaults to None.
        """
        super().__init__()
        self.margin = margin
        self.top_k_violations = top_k_violations

    def forward(self, scores: torch.tensor) -> torch.tensor:
        """
        Calculates the loss as defined in VSE++

        Args:
            scores (torch.tensor): The similarity matrix between all images and captions in a batch

        Returns:
            torch.tensor: The loss.
        """
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = (torch.eye(scores.size(0)) > .5).to(scores.device)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        # If top_k_violations is defined
        if self.top_k_violations:
            cost_s = cost_s.topk(self.top_k_violations, dim=1)[0]
            cost_im = cost_im.topk(self.top_k_violations, dim=0)[0]

        return cost_s.sum() + cost_im.sum()
    

class LxmertIRMatchingHead(nn.Module):
    """
    The matching head that sits on top of the pooled [CLS] token.
    """
    def __init__(self, config: LxmertConfig, num_labels: int) -> None:
        """
        Initialize the matching head

        Args:
            config ([type]): [description]
            num_labels (int): [description]
        """
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_labels),
        )
        
    def forward(self, hidden_states: torch.tensor) -> torch.tensor:
        """
        Forward of the matching head

        Args:
            hidden_states (torch.tensor): Hidden states of the [CLS] token.

        Returns:
            torch.tensor: Output tensor.
        """
        return self.logit_fc(hidden_states)


class LxmertForIRBCE(LxmertPreTrainedModel):
    """
    Binary classification formulation of text-image retrieval
    """
    def __init__(self, config: LxmertConfig) -> None:
        super().__init__(config)
        # Configuration
        self.config = config
        
        # Lxmert backbone
        self.lxmert = LxmertModel(config)
        
        self.match_head = LxmertIRMatchingHead(config, self.config.num_labels)
        
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
    ) -> Union[Tuple, LxmertForIROutput]:
        """
        Model forward for binary classification formulation

        Args:
            input_ids (torch.LongTensor, optional): The input ids from the tokenizer. Defaults to None.
            visual_feats (torch.tensor, optional): The pooled visual features from Faster R-CNN. Defaults to None.
            visual_pos (torch.tensor, optional): Normalized bounding box features from Faster R-CNN. Defaults to None.
            attention_mask (torch.tensor, optional): The attention mask from the tokenizer. Defaults to None.
            token_type_ids (torch.tensor, optional): What type of tokens (i.e. first or second sentence) from tokenizer. Defaults to None.
            visual_attention_mask (torch.tensor, optional): Custom visual attention mask. Defaults to None.
            inputs_embeds (torch.tensor, optional): From the tokenizer. Defaults to None.
            labels (torch.tensor, optional): Whether the paired image and text are a ground-truth pair. Defaults to None.
            output_attentions (bool, optional): Whether to output the attentions. Defaults to None.
            output_hidden_states (bool, optional): Whether to output the hidden states. Defaults to None.
            return_dict (bool, optional): Whether to output a tuple or a dictionary. Defaults to None.

        Returns:
            Union[Tuple, LxmertForIROutput]: The output of the forward
        """
        
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
            loss = self.loss(matching_score, labels.unsqueeze(1))

        if not return_dict:
            output = (matching_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        return LxmertForIROutput(
            loss=loss,
            matching_score=matching_score, 
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )
        

class LxmertForIRContrastive(LxmertPreTrainedModel):
    """
    Unfinished contrastive ranking version of LXMERT for text-image retrieval.
    """
    def __init__(self, config: LxmertForIRConfig) -> None:
        """
        Initialize a contrastive ranking LXMERT

        Args:
            config (LxmertForIRConfig): The configuration that additionally defines margin and top k violations.
        """
        super().__init__(config)
        # Configuration
        self.config = config
        
        # Lxmert backbone
        self.lxmert = LxmertModel(config)
        
        # Weight initialization
        self.init_weights()
        
        # Instead of BCE, use the contrastive loss as defined in `coco_ir.py`
        self.loss = ContrastiveLoss(margin=self.config.margin, top_k_violations=self.config.top_k_violations)
        
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
    ) -> Union[Tuple, LxmertForIROutput]:
        """
        Model forward for binary classification formulation

        Args:
            input_ids (torch.LongTensor, optional): The input ids from the tokenizer. Defaults to None.
            visual_feats (torch.tensor, optional): The pooled visual features from Faster R-CNN. Defaults to None.
            visual_pos (torch.tensor, optional): Normalized bounding box features from Faster R-CNN. Defaults to None.
            attention_mask (torch.tensor, optional): The attention mask from the tokenizer. Defaults to None.
            token_type_ids (torch.tensor, optional): What type of tokens (i.e. first or second sentence) from tokenizer. Defaults to None.
            visual_attention_mask (torch.tensor, optional): Custom visual attention mask. Defaults to None.
            inputs_embeds (torch.tensor, optional): From the tokenizer. Defaults to None.
            labels (torch.tensor, optional): Whether the paired image and text are a ground-truth pair. Defaults to None.
            output_attentions (bool, optional): Whether to output the attentions. Defaults to None.
            output_hidden_states (bool, optional): Whether to output the hidden states. Defaults to None.
            return_dict (bool, optional): Whether to output a tuple or a dictionary. Defaults to None.

        Returns:
            Union[Tuple, LxmertForIROutput]: The output of the forward
        """
        
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

        visual_output = lxmert_output[1]
        pooled_visual_output = visual_output.mean(dim=1)
        normalized_visual_output = F.normalize(pooled_visual_output, p=2, dim=1)
        pooled_textual_output = lxmert_output[2]
        normalized_textual_output = F.normalize(pooled_textual_output, p=2, dim=1)
        matching_score = (normalized_visual_output * normalized_textual_output).sum(dim=1)
        loss = None
        if labels is not None:
            match_matrix = torch.inner(normalized_visual_output, normalized_textual_output)
            loss = self.loss(match_matrix)

        if not return_dict:
            output = (matching_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        return LxmertForIROutput(
            loss=loss,
            matching_score=matching_score, 
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )
