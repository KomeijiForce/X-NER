def find_continuous_sequence(lst):
    res = []
    i, j = 0, 0
    while j < len(lst):
        if j + 1 < len(lst) and lst[j+1] == lst[j] + 1:
            j += 1
        else:
            res.append(lst[i:j+1])
            j += 1
            i = j
    return res

def parse_tag_chunk(_tags):
    tags = _tags
    
    matched = [idx for idx, tag in enumerate(tags) if tag != "O"]
    
    continuous_sequence = find_continuous_sequence(matched)
    
    start_ends = [(seq[0], seq[-1]+1, _tags[seq[0]]) for seq in continuous_sequence]
                
    return start_ends

def is_subtree(heads, start, end):

    ids = [idx for idx in range(start, end)]
    heads = [heads[idx] for idx in ids]
    overlap = [idx for idx in heads if idx not in ids]

    return len(overlap) == 1 and 0 not in heads

def noun_chunk_parse_ttag(sent, tags, ttags):
    words = sent.split()
    
    matched = [idx for idx, tag in enumerate(tags) if tag in ttags]
    
    continuous_sequence = find_continuous_sequence(matched)
    
    start_ends = [(seq[0], seq[-1]+1) for seq in continuous_sequence]
                
    return start_ends

def noun_chunk_parse(sent, tags):
                
    return noun_chunk_parse_ttag(sent, tags, ["NOUN", "PROPN"])