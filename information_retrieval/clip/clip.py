from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from transformers import CLIPModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling


@dataclass
class CLIPForIROutput(ModelOutput):
    """
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`return_loss` is :obj:`True`):
            Contrastive loss for image-text similarity.
        matching_score:(:obj:`torch.FloatTensor` of shape `(image_batch_size,)`):
            The diagonal entries of the `logits_per_image` attribute, which represents the image-text similarity
            scores for the actual pairing in the dataloader.
        logits_per_image:(:obj:`torch.FloatTensor` of shape :obj:`(image_batch_size, text_batch_size)`):
            The scaled dot product scores between :obj:`image_embeds` and :obj:`text_embeds`. This represents the
            image-text similarity scores.
        logits_per_text:(:obj:`torch.FloatTensor` of shape :obj:`(text_batch_size, image_batch_size)`):
            The scaled dot product scores between :obj:`text_embeds` and :obj:`image_embeds`. This represents the
            text-image similarity scores.
        text_embeds(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.CLIPTextModel`.
        image_embeds(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.CLIPVisionModel`.
        text_model_output(:obj:`BaseModelOutputWithPooling`):
            The output of the :class:`~transformers.CLIPTextModel`.
        vision_model_output(:obj:`BaseModelOutputWithPooling`):
            The output of the :class:`~transformers.CLIPVisionModel`.
    """

    loss: Optional[torch.FloatTensor] = None
    matching_score: torch.FloatTensor = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
    

class CLIPForIR(CLIPModel): 
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids = None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_loss = labels is not None
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        clip_output = super().forward(input_ids=input_ids, 
                                      pixel_values=pixel_values, 
                                      attention_mask=attention_mask, 
                                      position_ids=position_ids,
                                      return_loss=return_loss,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states,
                                      return_dict=return_dict)
        if not return_dict:
            matching_score = torch.diagonal(clip_output[0 if not return_loss else 1]).contiguous()
            return (clip_output[0], matching_score) if return_loss else matching_score
        
        matching_score = torch.diagonal(clip_output.logits_per_image).contiguous()
        
        return CLIPForIROutput(
            loss=clip_output.loss,
            matching_score=matching_score,
            logits_per_image=clip_output.logits_per_image,
            logits_per_text=clip_output.logits_per_text,
            text_embeds=clip_output.text_embeds,
            image_embeds=clip_output.image_embeds,
            text_model_output=clip_output.text_model_output,
            vision_model_output=clip_output.vision_model_output,
        )
