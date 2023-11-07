import json
from nltk import sent_tokenize

def read_raw(fname):
    texts = []
    for line in open(fname):
        line = line.strip()
        if len(line) > 0:
            texts.extend(sent_tokenize(line))
    return texts

def read_train(dataset):
    if dataset == "nerd":
        chunks = open(f"ner/nerd/supervised/train.txt").read().strip().split("\n\n")
        tags = open(f"ner/nerd/tags.txt").read().strip().split("\n")
        texts = [" ".join([line.split()[0] for line in chunk.split("\n")]) for chunk in chunks]

        ents = []

        for idx, chunk in enumerate(chunks):

            _tags = [line.split()[1] for line in chunk.split("\n")]
            start_ends = parse_tag_chunk(_tags)
            ents.extend([(idx, start_end) for start_end in start_ends])

    elif dataset == "ade":

        texts, ents = [], []

        _cnt = 0

        for idx in range(1, 10):
            items = json.load(open(f"ner/ade/test.ADE{idx}.json"))
            texts.extend([" ".join(item["tokens"]) for item in items])

            for _, item in enumerate(items):

                for ent in item['entities']:
                    s, e, l = ent

                    ents.append((_cnt, (s, e, l)))

                _cnt += 1

    elif dataset == "conll03":

        chunks = open("ner/conll03/train.txt").read().strip().split("\n\n")
        texts = [" ".join([line.split()[0] for line in chunk.split("\n")]) for chunk in chunks]

        ents = []

        for idx, chunk in enumerate(chunks):

            labels = [line.split()[3] for line in chunk.split("\n")]
            S = [idx for idx, label in enumerate(labels) if label.split("-")[0] == "S"]
            L_S = [label.split("-")[1] for idx, label in enumerate(labels) if label.split("-")[0] == "S"]
            B = [idx for idx, label in enumerate(labels) if label.split("-")[0] == "B"]
            E = [idx for idx, label in enumerate(labels) if label.split("-")[0] == "E"]
            L_B = [label.split("-")[1] for idx, label in enumerate(labels) if label.split("-")[0] == "B"]

            ents.extend([(idx, (s, s+1, l_s)) for s, l_s in zip(S, L_S)] + [(idx, (b, e+1, l_b)) for b, e, l_b in zip(B, E, L_B)])


    elif dataset == "semeval14":

        lines = open("ner/semeval14/train_triplets.txt").read().strip().split("\n")
        texts = [line.split("####")[0].strip().lower() for line in lines]
        _triplets = [eval(line.split("####")[1].strip()) for line in lines]

        ents = []

        for idx, triplets in enumerate(_triplets):

            for triplet in triplets:
                aspect, opinion, polarity = triplet
                polarity = polarity.lower()
                ents.append((idx, (aspect[0], aspect[-1]+1, "aspect")))
                ents.append((idx, (opinion[0], opinion[-1]+1, "opinion")))
                ents.append((idx, (opinion[0], opinion[-1]+1, polarity)))
                
    return texts, ents