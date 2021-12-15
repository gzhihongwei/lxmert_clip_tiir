import argparse
import json
import os

from pathlib import Path

import numpy as np

from transformers import CLIPProcessor

from information_retrieval.clip.data import CLIPRetrievalDataset
from information_retrieval.utils import DataTrainingArguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str, help="Path to the data directory that has COCO.")
    parser.add_argument('--prob_unaligned', required=False, default=0, type=float, help="Probability that the images for each caption are randomly sampled from the negative images.")
    parser.add_argument('--cross_image_eval', action='store_true', help="Perform cross image inference, i.e. each image with all texts from other images.")
    parser.add_argument('--eval_img_keys_file', required=False, default="", type=str, help="Image key tsv to select a subset of images for evaluation.")
    parser.add_argument('--evaluation_output_file', required=True, type=str, help="Where the output of a model was stored")
    parser.add_argument('--img_idxs', required=False, nargs="+", help="List of image query indices to consider in caption retrieval")
    parser.add_argument('--caption_idxs', required=False, nargs="+", help="List of caption query indices to consider in image retrieal")
    parser.add_argument('--output_file', required=False, default=None, type=str, help="Where to dump json")
    args = parser.parse_args()
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    test_dataset = CLIPRetrievalDataset(processor, args, "test", False)
    
    model_eval_output_file = args.evaluation_output_file
        
    model_eval_output = np.load(model_eval_output_file)
    
    num_to_sample = 40
    
    results = {}
    
    ###########################################################################################################
    # Image -> Text
    ###########################################################################################################
    i2t_logits = model_eval_output['logits'].T
    i2t_labels = model_eval_output['labels'].T
    i2t_ranks = model_eval_output['i2t_ranks']
    
    # Get the ground truth captions
    i2t_ground_truth = np.arange(5000)
    ground_truth_img_idxs = i2t_ground_truth // 5
    ground_truth_img_keys = np.array(list(map(lambda x: test_dataset.img_keys[x], ground_truth_img_idxs)))
    ground_truth_cap_idxs = i2t_ground_truth % 5
    ground_truth_captions = np.array(list(map(lambda x: test_dataset.captions[x[0]][x[1]], zip(ground_truth_img_keys, ground_truth_cap_idxs))))
    img_ids = np.array(test_dataset.img_keys)
    ground_truth_images = img_ids[ground_truth_img_idxs]
    
    if args.img_idxs is None:
        # Get the "hard" image queries that didn't have a ground truth caption retrieved @ 10
        hard_image_inds = (i2t_ranks >= 10).nonzero()[0]
        hard_image_inds = np.random.choice(hard_image_inds, size=num_to_sample, replace=False).tolist()
        # Get the "easy" image queries that retrieved a ground truth caption @ 1
        easy_image_inds = (i2t_ranks < 1).nonzero()[0]
        easy_image_inds = np.random.choice(easy_image_inds, size=num_to_sample, replace=False).tolist()

        # Get the indices in the form for the PyTorch dataset to get the ground_truth captions
        i2t_sorted_logits = (-i2t_logits).argsort(axis=1)[:, :10]
        ranked_captions = np.take(ground_truth_captions, i2t_sorted_logits)

        hard_image_ids = img_ids[hard_image_inds].tolist()
        hard_ground_truth_captions = [[caption for caption in ground_truth_captions[5*i:(5*i)+5]] for i in hard_image_inds]
        hard_retrieved_captions = ranked_captions[hard_image_inds].tolist()
        i2t_hard_queries = [{"query": img_id, "ground_truth": ground_truth, "retrieved": retrieved}
                            for img_id, ground_truth, retrieved in zip(hard_image_ids, hard_ground_truth_captions, hard_retrieved_captions)]

        easy_image_ids = img_ids[easy_image_inds].tolist()
        easy_ground_truth_captions = [[caption for caption in ground_truth_captions[5*i:(5*i)+5]] for i in easy_image_inds]
        easy_retrieved_caption = ranked_captions[easy_image_inds][:, 0].tolist()
        i2t_easy_queries = [{"query": img_idx, "ground_truth": ground_truth, "retrieved": retrieved}
                            for img_idx, ground_truth, retrieved in zip(easy_image_ids, easy_ground_truth_captions, easy_retrieved_caption)]

        results["i2t"] = {"hard": i2t_hard_queries, "easy": i2t_easy_queries}
    else:
        image_inds = list(map(int, args.img_idxs))
        # Get the indices in the form for the PyTorch dataset to get the ground_truth captions
        i2t_sorted_logits = (-i2t_logits).argsort(axis=1)[:, :10]
        ranked_captions = np.take(ground_truth_captions, i2t_sorted_logits)
        
        image_ids = img_ids[image_inds].tolist()
        retrieved_captions = ranked_captions[image_inds].tolist()
        
        results["i2t"] = [{"query": img_idx, "retrieved": retrieved}
                          for img_idx, retrieved in zip(image_ids, retrieved_captions)]
        
    
    ###########################################################################################################
    # Text -> Image
    ###########################################################################################################
    t2i_logits = model_eval_output['logits']
    t2i_labels = model_eval_output['labels']
    t2i_ranks = model_eval_output['t2i_ranks']
        
    if args.caption_idxs is None:

        # Get the "hard" caption queries that didn't retrieve the ground truth image @ 10
        hard_caption_inds = (t2i_ranks >= 10).nonzero()[0]
        hard_caption_inds = np.random.choice(hard_caption_inds, size=num_to_sample, replace=False).tolist()
        # Get the "easy" caption queries that retrieved the ground truth image @ 1
        easy_caption_inds = (t2i_ranks < 1).nonzero()[0]
        easy_caption_inds = np.random.choice(easy_caption_inds, size=num_to_sample, replace=False).tolist()

        ranked_images = (-t2i_logits).argsort(axis=1)[:, :10]

        hard_query_captions = ground_truth_captions[hard_caption_inds].tolist()
        hard_ground_truth_images = ground_truth_images[hard_caption_inds].tolist()
        hard_retrieved_images = ranked_images[hard_caption_inds].tolist()
        t2i_hard_queries = [{"query": query_caption, "ground_truth": img_idx, "retrieved": retrieved}
                            for query_caption, img_idx, retrieved in zip(hard_query_captions, hard_ground_truth_images, hard_retrieved_images)]

        easy_query_captions = ground_truth_captions[easy_caption_inds]
        easy_ground_truth_images = ground_truth_images[easy_caption_inds].tolist()
        t2i_easy_queries = [{"query": query_caption, "ground_truth": img_idx}
                            for query_caption, img_idx in zip(easy_query_captions, easy_ground_truth_images)]

        results["t2i"] = {"hard": t2i_hard_queries, "easy": t2i_easy_queries}
    else:
        caption_inds = list(map(int, args.caption_idxs))
        t2i_sorted_logits = (-t2i_logits).argsort(axis=1)[:, :10]
        ranked_images = np.take(img_ids, t2i_sorted_logits)
        
        query_captions = ground_truth_captions[caption_inds].tolist()
        retrieved_images = ranked_images[caption_inds].tolist()
        
        results["t2i"] = [{"query": query_caption, "retrieved": retrieved}
                          for query_caption, retrieved in zip(query_captions, retrieved_images)]
        
    output_path = f"{os.path.splitext(model_eval_output_file)[0]}.json" if args.output_file is None else args.output_file
    with open(output_path, "w") as f:
        json.dump(results, f)
