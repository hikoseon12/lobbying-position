import os
import csv
import time
import json
import copy
import backoff
import argparse
import subprocess
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict
from functools import partial
from multiprocessing import Pool
from openai import RateLimitError
from azure_openai_api import inference_azure
from utils import open_csv, save_csv


@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def get_gpt_output(gpt_input, **kwargs):
    gpt_output = inference_azure(gpt_input, **kwargs)
    return gpt_output


def get_instruction(bill_no):
    instruction=f"""Classify the given text that explicitly describes lobbying activities for bill {bill_no} into one of the five types: 'Support', 'Oppose', 'Amend', 'Monitor', or 'Mention' without explanation."""
    return instruction


# get input prompt
def get_gpt_prompt(df, **kwargs):
    gpt_input = ''
    try:
        paragraph = df['paragraph']
        bill_no = df['bill_no']
        short_title = str(df['short_title'])
        official_title = df['official_title']
        bill_no = str(df['bill_no'])
        bill_no = bill_no.replace('hr', 'H.R.')
        bill_no = bill_no.replace('s', 'S.')
        instruction = get_instruction(bill_no)
        instruction = f"{instruction}"
        titles = ''

        if short_title == 'nan':
            titles = f"{bill_no} official title: {official_title}"
        else:
            titles = f"{bill_no} short title: {short_title}\n{bill_no} official title: {official_title}"
        gpt_input = f"{instruction}\n{titles}\nText:{paragraph}\nAnswer:"

    except Exception as e:
        print(f'Exception ERROR: {e}')
        gpt_input = 'ERROR'
    return instruction, gpt_input


# get gpt result
def get_chatgpt_result(df, **kwargs):
    results = []
    with open(f"{kwargs['output_dir']}/{kwargs['output_name']}", "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i,row in tqdm(df.iterrows(),total=df.shape[0]):
            instruction, gpt_prompt = get_gpt_prompt(row, **kwargs)
            if 'ERROR' == gpt_prompt:
                continue

            gpt_output = get_gpt_output(gpt_prompt, **kwargs)
            result_row = row.tolist() + [gpt_output]
            writer.writerow(result_row)
            results.append(result_row)
            
            with open(f"{kwargs['output_dir']}/{kwargs['output_name'].replace('.csv','')}.pkl", "wb") as pkl_file:
                pkl.dump(results, pkl_file)


# run multiprocessing
def chatgpt_multiprocessing(df, **kwargs):
    df_split = np.array_split(df, kwargs['num_cores'])
    with Pool(kwargs['num_cores']) as pool:
        pool.map(partial(get_chatgpt_result, **kwargs), df_split)
    print('Pooling done...')


def main(args):
    df_column = ['paragraph','bill_id','bill_no','short_title','official_title']
    df = open_csv(f"{args.input_path}")
    df = df[df_column]

    output_dir_name = f"{args.output_dir}/{args.output_name}"
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    column_list = df.columns.tolist() + ['label']
    print(f"LLM annotation output path : {output_dir_name}")
    with open(f"{output_dir_name}", 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(column_list)

    chatgpt_multiprocessing(
        df,
        **vars(args)
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api_version", type=str, required=True)
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--azure_endpoint", type=str, default='')
    parser.add_argument("--num_cores", type=int, default=1)

    args = parser.parse_args()
    return args 
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)