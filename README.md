# X-NER
Official Implementation for "Less than One-shot: Named Entity Recognition via Extremely Weak Supervision"

## What is this for?
You can use the scripts in this repo to discover semantically similar entities with only one entity span and its category name!

Try a mining on a short Touhou Wiki using the script ``run_annotate.sh``:
```
bash run_annotate.sh
```

With the default setups, you will find the mining result in ``PER.annotate.csv`` with mined person names and their distance to the seed span.

You can also adjust the script ``run_annotate.sh`` to mine on your own texts:
```
python annotate.py\
       --fname [YOUR CORPUS (TXT)]\
       --ent_label [LABEL]\
       --query [ENTITY NAME]\
       --description "The [LABEL NAME] name: X" or [MORE DETAILED DESCRIPTION]\
       --device_idx [YOUR DEVICE]\
       --output_file [YOUR FILE (CSV)]\
```

For datasets in format like CoNLL03, please run the ``run_mine.sh`` script:

```
bash run_mine.sh
```

You can also adjust it for your needs:

```
python main.py\
       --dataset [DATASET PATH]\
       --ent_label [LABEL]\
       --query [ENTITY NAME]\
       --description "The [LABEL NAME] name: X" or [MORE DETAILED DESCRIPTION]\
       --device_idx [YOUR DEVICE]\
       --output_file [YOUR FILE (CSV)]\
```
