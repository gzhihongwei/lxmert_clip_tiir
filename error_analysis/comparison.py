import argparse
import json

import numpy as np

from transformers import CLIPProcessor

from information_retrieval.clip.data import CLIPRetrievalDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str, help="Path to the data directory that has COCO.")
    parser.add_argument('--prob_unaligned', required=False, default=0, type=float, help="Probability that the images for each caption are randomly sampled from the negative images.")
    parser.add_argument('--cross_image_eval', action='store_true', help="Perform cross image inference, i.e. each image with all texts from other images.")
    parser.add_argument('--eval_img_keys_file', required=False, default="", type=str, help="Image key tsv to select a subset of images for evaluation.")
    parser.add_argument('--evaluation_output_file', required=True, type=str, help="Where the output of a model was stored")
    args = parser.parse_args()
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    test_dataset = CLIPRetrievalDataset(processor, args, "test", False)
    
    first_eval_output_file, second_eval_output_file = "clip_1k_test.npz", "lxmert_1k_test.npz"
        
    first_eval_output = np.load(first_eval_output_file)
    second_eval_output = np.load(second_eval_output_file)
    
    results = {}
    
    ###########################################################################################################
    # Image -> Text
    ###########################################################################################################
    first_i2t_ranks = first_eval_output['i2t_ranks']
    second_i2t_ranks = second_eval_output['i2t_ranks']
    
    # Get the image indices where the first model did better than the second model
    first_image_inds = ((first_i2t_ranks < 10) & (first_i2t_ranks < second_i2t_ranks) & ((second_i2t_ranks - first_i2t_ranks) > 9)).nonzero()[0].tolist()
    
    print(f"Number of images where the first model performed better than the second: {len(first_image_inds)}")
    print(list(zip(first_i2t_ranks[first_image_inds], second_i2t_ranks[first_image_inds])))
    
    # Get the ground truth captions
    i2t_ground_truth = np.arange(5000)
    ground_truth_img_idxs = i2t_ground_truth // 5
    ground_truth_img_keys = np.array(list(map(lambda x: test_dataset.img_keys[x], ground_truth_img_idxs)))
    ground_truth_cap_idxs = i2t_ground_truth % 5
    ground_truth_captions = np.array(list(map(lambda x: test_dataset.captions[x[0]][x[1]], zip(ground_truth_img_keys, ground_truth_cap_idxs))))
    
    img_ids = np.array(test_dataset.img_keys)
    
    first_image_ids = img_ids[first_image_inds].tolist()
    first_text_ranks = first_i2t_ranks[first_image_inds].tolist()
    second_text_ranks = second_i2t_ranks[first_image_inds].tolist()
    results["i2t"] = [{"query": img_id, "clip": clip_rank, "lxmert": lxmert_rank}
                      for img_id, clip_rank, lxmert_rank in zip(first_image_ids, first_text_ranks, second_text_ranks)]
    
    ###########################################################################################################
    # Text -> Image
    ###########################################################################################################
    first_t2i_ranks = first_eval_output['t2i_ranks']
    second_t2i_ranks = second_eval_output['t2i_ranks']
    
    # Get the image indices where the first model did better than the second model
    first_caption_inds = ((first_t2i_ranks < 10) & (first_t2i_ranks < second_t2i_ranks) & ((second_t2i_ranks - first_t2i_ranks) > 9)).nonzero()[0].tolist()
    
    print(f"Number of captions where the first model performed better than the second: {len(first_caption_inds)}")
    print(list(zip(first_t2i_ranks[first_caption_inds], second_t2i_ranks[first_caption_inds])))
    
    # Get the ground truth captions
    first_captions = ground_truth_captions[first_caption_inds].tolist()
    first_image_ranks = first_t2i_ranks[first_caption_inds].tolist()
    second_image_ranks = second_t2i_ranks[first_caption_inds].tolist()
    
    results["t2i"] = [{"query": caption, "clip": clip_rank, "lxmert": lxmert_rank}
                      for caption, clip_rank, lxmert_rank in zip(first_captions, first_image_ranks, second_image_ranks)]
    
    with open("comparison.json", "w") as f:
        json.dump(results, f)
