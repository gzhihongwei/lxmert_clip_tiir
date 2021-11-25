import json
import os

from pathlib import Path

import numpy as np

from transformers import CLIPProcessor, HfArgumentParser

from information_retrieval.clip.data import CLIPRetrievalDataset
from information_retrieval.utils import DataTrainingArguments


if __name__ == "__main__":
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    test_dataset = CLIPRetrievalDataset(processor, data_args, "test", False)
    
    model_eval_output_file = data_args.evaluation_output_file
    model_eval_output = np.load(model_eval_output_file)
    
    num_to_sample = 40
    
    results = {}
    
    ###########################################################################################################
    # Image -> Text
    ###########################################################################################################
    i2t_logits = model_eval_output['logits'].T
    i2t_labels = model_eval_output['labels'].T
    i2t_ranks = model_eval_output['i2t_ranks']
    
    # Get the "hard" image queries that didn't have a ground truth caption retrieved @ 10
    hard_image_inds = (i2t_ranks >= 10).nonzero()[0]
    hard_image_inds = np.random.choice(hard_image_inds, size=num_to_sample, replace=False)
    # Get the "easy" image queries that retrieved a ground truth caption @ 1
    easy_image_inds = (i2t_ranks < 1).nonzero()[0]
    easy_image_inds = np.random.choice(easy_image_inds, size=num_to_sample, replace=False)
    
    # Get the ground truth captions
    i2t_ground_truth = np.arange(5000)
    ground_truth_img_idxs = i2t_ground_truth // 5
    ground_truth_img_keys = np.array(map(lambda x: test_dataset.img_keys[x], ground_truth_img_idxs))
    ground_truth_cap_idxs = i2t_ground_truth % 5
    ground_truth_captions = np.array(map(lambda x: test_dataset.captions[x[0]][x[1]], zip(ground_truth_img_keys, ground_truth_cap_idxs)))
    
    # Get the indices in the form for the PyTorch dataset to get the ground_truth captions
    i2t_sorted_logits = (-i2t_logits).argsort(axis=1)[:, :10]
    ranked_captions = np.take(ground_truth_captions, i2t_sorted_logits)
    
    hard_ground_truth_captions = [[caption for caption in ground_truth_captions[5*i:(5*i)+5]] for i in hard_image_inds]
    hard_retrieved_captions = ranked_captions[hard_image_inds]
    i2t_hard_queries = [{"query": img_idx, "ground_truth": ground_truth, "retrieved": retrieved}
                        for img_idx, ground_truth, retrieved in zip(hard_image_inds, hard_ground_truth_captions, hard_retrieved_captions)]
    
    easy_ground_truth_captions = [[caption for caption in ground_truth_captions[5*i:(5*i)+5]] for i in easy_image_inds]
    easy_retrieved_caption = ranked_captions[easy_image_inds][:, 0]
    i2t_easy_queries = [{"query": img_idx, "ground_truth": ground_truth, "retrieved": retrieved}
                        for img_idx, ground_truth, retrieved in zip(easy_image_inds, easy_ground_truth_captions, easy_retrieved_caption)]
    
    results["i2t"] = {"hard": i2t_hard_queries, "easy": i2t_easy_queries}
    
    ###########################################################################################################
    # Text -> Image
    ###########################################################################################################
    t2i_logits = model_eval_output['logits']
    t2i_labels = model_eval_output['labels']
    t2i_ranks = model_eval_output['t2i_ranks']
    
    # Get the "hard" caption queries that didn't retrieve the ground truth image @ 10
    hard_caption_inds = (t2i_ranks >= 10).nonzero()[0]
    hard_caption_inds = np.random.choice(hard_caption_inds, size=num_to_sample, replace=False)
    # Get the "easy" caption queries that retrieved the ground truth image @ 1
    easy_caption_inds = (t2i_ranks < 1).nonzero()[0]
    easy_caption_inds = np.random.choice(easy_caption_inds, size=num_to_sample, replace=False)
    
    ground_truth_images = ground_truth_img_idxs
    
    ranked_images = (-t2i_logits).argsort(axis=1)[:, :10]
    
    hard_query_captions = ground_truth_captions[hard_caption_inds]
    hard_ground_truth_images = ground_truth_images[hard_caption_inds]
    hard_retrieved_images = ranked_images[hard_caption_inds]
    t2i_hard_queries = [{"query": query_caption, "ground_truth": img_idx, "retrieved": retrieved}
                        for query_caption, img_idx, retrieved in zip(hard_query_captions, hard_ground_truth_images, hard_retrieved_images)]
    
    easy_query_captions = ground_truth_captions[easy_caption_inds]
    easy_ground_truth_images = ground_truth_images[easy_caption_inds]
    t2i_easy_queries = [{"query": query_caption, "ground_truth": img_idx}
                        for query_caption, img_idx in zip(easy_query_captions, easy_ground_truth_images)]
    
    results["t2i"] = {"hard": t2i_hard_queries, "easy": t2i_easy_queries}
    
    with open(f"{os.path.basename(model_eval_output_file)}.json", "w") as f:
        json.dump(results, f)
