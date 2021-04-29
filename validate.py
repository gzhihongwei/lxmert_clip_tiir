from __future__ import print_function

import numpy as np
import torch


def encode_data(tokenizer, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    all_input_ids = None
    all_token_type_ids = None
    all_attention_masks = None
    all_visual_feats = None
    all_visual_pos = None
    
    for i, (captions, visual_feats, visual_pos, indices) in enumerate(data_loader):
        tokenized = tokenizer(list(captions), padding='max_length', return_tensors='pt')
        input_ids, token_type_ids, attention_masks = tokenized['input_ids'], tokenized['token_type_ids'], tokenized['attention_mask']
        # initialize the numpy arrays given the size of the embeddings
        if all_input_ids is None:
            all_input_ids = np.zeros((len(data_loader.dataset), input_ids.size(1)))
            all_token_type_ids = np.zeros((len(data_loader.dataset), token_type_ids.size(1)))
            all_attention_masks = np.zeros((len(data_loader.dataset), attention_masks.size(1)))
            all_visual_feats = np.zeros((len(data_loader.dataset), *visual_feats.shape[1:]))
            all_visual_pos = np.zeros((len(data_loader.dataset), *visual_pos.shape[1:]))

        # preserve the embeddings by copying from gpu and converting to numpy
        all_input_ids[indices] = input_ids.cpu().numpy().copy()
        all_token_type_ids[indices] = token_type_ids.cpu().numpy().copy()
        all_attention_masks[indices] = attention_masks.cpu().numpy().copy()
        all_visual_feats[indices] = visual_feats.cpu().numpy().copy()
        all_visual_pos[indices] = visual_pos.cpu().numpy().copy()


        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'.format(i+1, len(data_loader)))

    return all_input_ids, all_token_type_ids, all_attention_masks, all_visual_feats, all_visual_pos


def rank_captions(model, all_input_ids, all_token_type_ids, 
                  all_attention_masks, query_visual_feats, query_visual_pos):
    model.eval()
    
    ranks = torch.zeros((100, all_input_ids.size(0))).cuda()
    
    for i, (visual_feats, visual_pos) in enumerate(zip(query_visual_feats, query_visual_pos)):
        scores = torch.zeros(all_input_ids.size(0)).cuda()
        
        for j in range(0, all_input_ids.size(0), 100):
            with torch.no_grad():
                batch_scores = model(all_input_ids[j:j+100],
                                     visual_feats.expand(100, *visual_feats.shape),
                                     visual_pos.expand(100, *visual_pos.shape),
                                     all_attention_masks[j:j+100],
                                     all_token_type_ids[j:j+100])
                scores[j:j+100] = batch_scores.matching_score.flatten()
                
        ranks[i] = scores
    
    model.train()
    
    return ranks


def rank_images(model, query_input_ids, query_token_type_ids, query_attention_masks,
                all_visual_feats, all_visual_pos):
    model.eval()
    
    ranks = torch.zeros((100, all_visual_feats.size(0))).cuda()
    
    for i, (input_ids, token_type_ids, attention_mask) in enumerate(zip(query_input_ids, query_token_type_ids, query_attention_masks)):
        scores = torch.zeros(all_visual_feats.size(0)).cuda()
        
        for j in range(0, all_visual_feats.size(0), 100):
            with torch.no_grad():
                batch_scores = model(input_ids.expand(100, *input_ids.shape),
                                     all_visual_feats[j:j+100],
                                     all_visual_pos[j:j+100],
                                     attention_mask.expand(100, *attention_mask.shape),
                                     token_type_ids.expand(100, *token_type_ids.shape))
                scores[j:j+100] = batch_scores.matching_score.flatten()
                
        ranks[i] = scores

    model.train()
    
    return ranks


def i2t(model, all_input_ids, all_token_type_ids, all_attention_masks,
        all_visual_feats, all_visual_pos, logging, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(all_input_ids.shape[0] / 5)
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    
    for index in range(npts):

        # Compute scores
        bs = 100
        
        if index % bs == 0:
            logging('On batch {}/{}'.format((index // bs) + 1, npts // bs))
            mx = min(all_visual_feats.shape[0], 5 * (index + bs))
            visual_feats = all_visual_feats[5 * index:mx:5]
            visual_pos = all_visual_pos[5 * index:mx:5]
            d2 = rank_captions(model,
                               torch.Tensor(all_input_ids).long().cuda(),
                               torch.Tensor(all_token_type_ids).long().cuda(),
                               torch.Tensor(all_attention_masks).cuda(),
                               torch.Tensor(visual_feats).cuda(),
                               torch.Tensor(visual_pos).cuda()).cpu().numpy()
            
        d = d2[index % bs]
        
        inds = np.argsort(-d)
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(model, all_input_ids, all_token_type_ids, all_attention_masks,
        all_visual_feats, all_visual_pos, logging, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(all_input_ids.shape[0] / 5)
    all_visual_feats = all_visual_feats[::5]
    all_visual_pos = all_visual_pos[::5]

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Compute scores
        bs = 100
        
        if 5 * index % bs == 0:
            logging('On batch {}/{}'.format((5 * index // bs) + 1, 5 * npts // bs))
            mx = min(all_input_ids.shape[0], 5 * index + bs)
            # Get query captions
            query_input_ids = all_input_ids[5 * index:mx]
            query_token_type_ids = all_token_type_ids[5 * index:mx]
            query_attention_masks = all_attention_masks[5 * index:mx]
            d2 = rank_images(model,
                             torch.Tensor(query_input_ids).long().cuda(),
                             torch.Tensor(query_token_type_ids).long().cuda(),
                             torch.Tensor(query_attention_masks).cuda(),
                             torch.Tensor(all_visual_feats).cuda(),
                             torch.Tensor(all_visual_pos).cuda()).cpu().numpy()

        d = d2[(5 * index) % bs:(5 * index) % bs + 5]
        inds = np.zeros(d.shape)
        
        for i in range(len(inds)):
            inds[i] = np.argsort(-d[i])
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
    