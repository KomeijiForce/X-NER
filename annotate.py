import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import defaultdict
import logging
from utils import parse_tag_chunk
from data import read_raw
from constant import stops, nums, words2sent
from mining import fastndd_search
import argparse


def main(args):
    setup_logging()
    device, tokenizer, model = load_model_and_tokenizer(args)
    description = create_description(args, tokenizer)
    dfs = mine_entities(args, texts, tokenizer, model, device, description)
    save_results(dfs, texts, args.output_file)


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def load_model_and_tokenizer(args):
    logger.info(f"Building model & tokenizer from path '{args.model_path}' to device '{args.device_idx}'...")
    device = torch.device(args.device_idx)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device)
    return device, tokenizer, model


def create_description(args, tokenizer):
    if args.description == "MASK CONTEXT":
        logger.info(f"Using the mask context...")
        description = f"{tokenizer.mask_token} X"
    else:
        description = args.description
        assert ("X" in description)
    return description


def mine_entities(args, texts, tokenizer, model, device, description):
    logger.info(
        f"Mining {args.ent_label} entities on the '{args.fname}' with seed span '{args.query}' and description '{description}'...")
    dfs = []
    bar = tqdm(enumerate(texts), total=len(texts))

    for cnt, text in bar:
        df = fastndd_search(model, tokenizer, text, args.query, description, args.ent_label, device,
                            max_len=args.max_len, bs=args.bs, temp=args.temp, cnt=cnt)
        if df is not None:
            dfs.append(df)

    return dfs


def save_results(dfs, texts, output_file):
    logger.info(f"Saving the mining results to '{output_file}'...")
    df_cat = pd.concat(dfs).sort_values("Score")
    df_cat["context"] = df_cat["Tuple"].apply(lambda x:texts[x[0]])
    df_cat.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for divergence-based entity mining.")
    parser.add_argument("--model_path", type=str, default="roberta-large")
    parser.add_argument("--fname", type=str, default="wiki/example.txt")
    parser.add_argument("--ent_label", type=str, default="PER")
    parser.add_argument("--query", type=str, default="Komeiji Koishi")
    parser.add_argument("--description", type=str, default="MASK CONTEXT")
    parser.add_argument("--device_idx", type=str, default="cuda:1")
    parser.add_argument("--output_file", type=str, default="DEFAULT")
    parser.add_argument("--max_len", type=int, default=3)
    parser.add_argument("--temp", type=float, default=2)
    parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()

    if args.output_file == "DEFAULT":
        args.output_file = f"{args.ent_label}.annotate.csv"

    logger = logging.getLogger(__name__)
    texts = read_raw(args.fname)
    main(args)
