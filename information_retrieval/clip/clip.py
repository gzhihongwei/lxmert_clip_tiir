from dataclasses import dataclass

import torch

from transformers import CLIPModel, CLIPOutput


@dataclass
class CLIPForIROutput(CLIPOutput):
    matching_score: torch.FloatTensor = None
    

class CLIPForIR(CLIPModel): 
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids = None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        clip_output = super().forward(input_ids=input_ids, 
                                      pixel_values=pixel_values, 
                                      attention_mask=attention_mask, 
                                      position_ids=position_ids,
                                      return_loss=return_loss,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states,
                                      return_dict=return_dict)
        if not return_dict:
            matching_score = torch.diagonal(clip_output[0 if not return_loss else 1])
            return (clip_output[0], matching_score) + clip_output[1:] if return_loss else (matching_score,) + clip_output
        
        matching_score = torch.diagonal(clip_output.logits_per_image)
        
        return CLIPForIROutput(
            loss=clip_output.loss,
            matching_score=matching_score,
            logits_per_image=clip_output.logits_per_image,
            logits_per_text=clip_output.logits_per_text,
            text_embeds=clip_output.text_embeds,
            image_embeds=clip_output.image_embeds,
            text_model_output=clip_output.text_model_outputs,
            vision_model_output=clip_output.vision_model_output,
        )