import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from constant import stops, nums, words2sent

def create_query(tokenizer, sent, new_sent, template, device, fast=False):
    with torch.no_grad():
        ids = tokenizer(sent, return_tensors="pt").input_ids.to(device)
        new_ids = tokenizer(new_sent, return_tensors="pt").input_ids.to(device)
        template_ids = tokenizer(template, return_tensors="pt").input_ids.to(device)

        for idx in range(min(ids.shape[1], template_ids.shape[1])):
            if ids[0, idx] != template_ids[0, idx]:
                break

        for jdx in range(1, 1+min(ids.shape[1], template_ids.shape[1])):
            if ids[0, -jdx] != template_ids[0, -jdx]:
                break
            
        idss_masked, new_idss_masked = [], []

        for idx in range(2 if not fast else idx, idx+1):
            ids_lmasked, new_ids_lmasked = deepcopy(ids), deepcopy(new_ids)
            ids_lmasked[0, idx-1], new_ids_lmasked[0, idx-1] = tokenizer.mask_token_id, tokenizer.mask_token_id
            idss_masked.append(ids_lmasked)
            new_idss_masked.append(new_ids_lmasked)

        for jdx in range(3 if not fast else jdx, jdx+1):
            ids_rmasked, new_ids_rmasked = deepcopy(ids), deepcopy(new_ids)
            ids_rmasked[0, -(jdx-1)], new_ids_rmasked[0, -(jdx-1)] = tokenizer.mask_token_id, tokenizer.mask_token_id
            idss_masked.append(ids_rmasked)
            new_idss_masked.append(new_ids_rmasked)
            
        idss_masked = torch.cat(idss_masked, 0)
        new_idss_masked = torch.cat(new_idss_masked, 0)

        return idss_masked, new_idss_masked
    
def fastndd_search(model, tokenizer, text, query, description, ent_label, device, max_len=3, bs=128, temp=2., cnt=0):
    with torch.no_grad():
        words = text.split()
        words_lower = [word.lower() for word in words]
        spans, scores = [], []

        start_ends = []
        max_len = min(max_len, len(words))

        start_ends = [(idx, idx+n_word) for n_word in range(max_len, 0, -1) for idx in range(len(words)-n_word+1-(1 if words[-1] == "." else 0))]
        start_ends = [start_end for start_end in start_ends if not any([stop in words_lower[start_end[0]:start_end[1]] for stop in stops])]
        start_ends = [start_end for start_end in start_ends if not any([num in " ".join(words_lower[start_end[0]:start_end[1]]) for num in nums])]

        if len(start_ends) == 0:
            return None

        ids = []
        tups = []
        idss_masked_batch, new_idss_masked_batch = [], []
        cut_points = []

        for start_end in start_ends:

            cut_point = 0

            start, end = start_end
            new_words = words[:start] + [query] + words[end:]
            template_words = words[:start] + ["X"] + words[end:]

            sent = words2sent(words)
            new_sent = words2sent(new_words)
            template = words2sent(template_words)

            span = words2sent(words[start:end])
            spans.append(span)

            tups.append((cnt, (start, end, ent_label)))

            idss_masked, new_idss_masked = create_query(tokenizer, sent, new_sent, template, device, fast=True) 
            idss_masked_batch.append(idss_masked), new_idss_masked_batch.append(new_idss_masked)
            cut_point += idss_masked.shape[0]
            idss_masked, new_idss_masked = create_query(tokenizer, description.replace("X", query), description.replace("X", span), description, device, fast=False)
            idss_masked_batch.append(idss_masked), new_idss_masked_batch.append(new_idss_masked)
            cut_point += idss_masked.shape[0]

            cut_points.append(cut_point)


        X = np.sum([idss_masked.shape[0] for idss_masked in idss_masked_batch]) + np.sum([new_idss_masked.shape[0] for new_idss_masked in new_idss_masked_batch])
        Y = max([np.amax([idss_masked.shape[1] for idss_masked in idss_masked_batch]), np.amax([new_idss_masked.shape[1] for new_idss_masked in new_idss_masked_batch])])

        all_idss_masked = torch.zeros((X, Y)).long().to(device)
        all_attn_mask = torch.zeros((X, Y)).long().to(device)

        x = 0
        for idss_masked in idss_masked_batch:
            all_idss_masked[x:x+idss_masked.shape[0], :idss_masked.shape[1]] = idss_masked
            all_attn_mask[x:x+idss_masked.shape[0], :idss_masked.shape[1]] = 1
            x += idss_masked.shape[0]

        for new_idss_masked in new_idss_masked_batch:
            all_idss_masked[x:x+new_idss_masked.shape[0], :new_idss_masked.shape[1]] = new_idss_masked
            all_attn_mask[x:x+new_idss_masked.shape[0], :new_idss_masked.shape[1]] = 1
            x += new_idss_masked.shape[0]

        mask = all_idss_masked.eq(tokenizer.mask_token_id)
        probs = torch.cat([(model(all_idss_masked[idx:idx+bs], all_attn_mask[idx:idx+bs]).logits[mask[idx:idx+bs]] / temp).softmax(-1) for idx in range(0, X, bs)], 0)

        probs, new_probs = probs[:X//2], probs[X//2:]

        divs = (probs * (probs/new_probs).log()).sum(-1)

        c = 0
        scores = []

        for cut_point in cut_points:

            score = divs[c:c+cut_point].mean().item()

            c += cut_point

            scores.append(score)

        return pd.DataFrame({
            "Span":spans,
            "Score":scores,
            "Tuple":tups,
        })