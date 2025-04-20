import os
import re
import string
import argparse
import numpy as np
import pandas as pd
from utils import open_csv, save_csv


def check_label(df):
    labels = ['Support','Oppose','Amend','Monitor','Mention']
    df['label'] = df['label'].apply(lambda x: x.strip(string.punctuation))
    df = df[df['label'].isin(labels)]
    return df


def convert_label(df):    
    df['label'] = df['label'].apply(lambda x: 1 if x=='Support' else x)
    df['label'] = df['label'].apply(lambda x: 2 if x=='Oppose' else x)
    df['label'] = df['label'].apply(lambda x: 3 if x=='Amend' else x)
    df['label'] = df['label'].apply(lambda x: 4 if x=='Monitor' else x)
    df['label'] = df['label'].apply(lambda x: 0 if (x=='Mention') else x)
    return df


def filter_none_label_df(df):
    df = df[df['label'].isin([1,2,3,4])]
    return df


def filter_bill_type_df(df):
    df['bill_type'] = df['bill_id'].apply(lambda x:re.split(r'(\d+)', x)[0])
    df = df[df['bill_type'].isin(['hr','s'])]
    return df


def merge_label_info_df(df, report_info_df):
    df = df[['paragraph','bill_id','short_title','label']]
    merged_df = pd.merge(report_info_df, df[['paragraph','bill_id','short_title','label']])
    no_dup_df = merged_df[['lob_id','bill_id','label']].drop_duplicates(subset=['lob_id','bill_id'])
    no_dup_df = merged_df[~merged_df.duplicated(subset=['lob_id','bill_id','label'],keep=False)]
    no_dup_df = no_dup_df[~no_dup_df.duplicated(subset=['lob_id','bill_id'],keep=False)]
    return merged_df


def process_llm_label_df(args, raw_all_llm_df, report_info_df):
    pre_all_llm_df = check_label(raw_all_llm_df)
    pre_all_llm_df = convert_label(pre_all_llm_df)
    pre_all_llm_df = filter_none_label_df(pre_all_llm_df)
    pre_all_llm_df = filter_bill_type_df(pre_all_llm_df)
    
    all_llm_df = pre_all_llm_df.drop_duplicates(subset=['paragraph','bill_id','label'])
    all_llm_df = all_llm_df.drop_duplicates(subset=['paragraph','bill_id'])
    llm_bill_position_final_df = merge_label_info_df(all_llm_df, report_info_df)
    return llm_bill_position_final_df


def main(args):
    df = open_csv(f"{args.output_dir}/{args.output_name}")
    report_info_df = open_csv(f"{args.report_df_path}")
    llm_bill_position_final_df = process_llm_label_df(args, df, report_info_df)
    print(f"save df: f'{args.output_dir}/{args.final_output_name}")
    save_csv(f'{args.output_dir}/{args.final_output_name}',llm_bill_position_final_df)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_df_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--final_output_name", type=str, required=True)

    args = parser.parse_args()
    return args 
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)