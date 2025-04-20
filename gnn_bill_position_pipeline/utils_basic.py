import os
import pickle as pkl
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from termcolor import cprint


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def open_csv(data_path):
    data_file = pd.read_csv(data_path)
    return data_file


def save_csv(save_data_path, data):
    data.to_csv(save_data_path, index=False)


def open_pkl(data_path, verbose=True):
    with open(data_path, 'rb') as f:
        if verbose:
            cprint(f"Load from {data_path}", "green")
        data_file = pkl.load(f)
    return data_file


def save_pkl(save_data_path, data, verbose=True):
    with open(save_data_path, 'wb') as f:
        if verbose:
            cprint(f"Save at {save_data_path}", "blue")
        pkl.dump(data, f)


def times_100(x):
    return x * 100


def get_df_from_dict(df_dict):
    df = pd.DataFrame(df_dict).transpose()
    df['precision'] = df['precision'].apply(times_100)
    df['recall'] = df['recall'].apply(times_100)
    df['f1-score'] = df['f1-score'].apply(times_100)
    return df


def save_info_result_dict(output_path, save_file_name, meta_info_dict, result_dict_list, confusion_m_list, args):
    (t_result_dict, v_result_dict, test_result_dict) = result_dict_list
    (t_confusion_m, v_confusion_m, test_confusion_m) = confusion_m_list

    dict_file_name = f'{save_file_name}_emb{args.emb_size}_lr{args.lr}_lam{args.lam}.pkl'
    dict_path_name = os.path.join(output_path, dict_file_name)
    if not os.path.isfile(dict_path_name):  # create info & result dict
        info_result_dict = defaultdict(dict)
        best_result_dict = defaultdict(list)
    else:
        info_result_dict = open_pkl(dict_path_name)
        best_result_dict = info_result_dict['result_dict']

    best_result_dict['nth'].append(args.nth)
    # best_result_dict['best_epoch'].append(best_epoch)
    best_result_dict['t_result_dict'].append(get_df_from_dict(t_result_dict))
    best_result_dict['v_result_dict'].append(get_df_from_dict(v_result_dict))
    best_result_dict['test_result_dict'].append(get_df_from_dict(test_result_dict))
    best_result_dict['t_confusion_m'].append(t_confusion_m)
    best_result_dict['v_confusion_m'].append(v_confusion_m)
    best_result_dict['test_confusion_m'].append(test_confusion_m)

    info_result_dict['info_dict'] = meta_info_dict
    info_result_dict['result_dict'] = best_result_dict
    save_pkl(dict_path_name, info_result_dict)
    return
