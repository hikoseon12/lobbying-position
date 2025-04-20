import os
import numpy as np
import pandas as pd

def open_csv(data_path):
    data_file = pd.read_csv(data_path)
    return data_file


def save_csv(save_data_path, data):
    data.to_csv(save_data_path, index=False)
