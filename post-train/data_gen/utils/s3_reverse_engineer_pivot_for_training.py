#####################
### 2025-07-24:  create a script to select suitable tables to reverse engineer regular relational tables in PBI cases, to make them "unrelational", for training data generation
#####################



import os
import openai
from openai import AzureOpenAI

print(openai.__version__)
import glob
import json
import pandas as pd
import pickle
import time
import sys
import random
import numpy as np


def find_leftmost_key_cols(df, most_cols = 3):
    # from i = 0 to most_cols, check if the first i cols are all unique (and can therefore be key cols)
    for i in range(1, most_cols + 1):
        # check if first i cols val combination are unique
        if df.iloc[:, :i].drop_duplicates().shape[0] == df.shape[0]:
            return i
    return 0

# check if a df is suitable to reverse-engineer using pivot
def re_pivot(df, use_aggressive_pivot=False):
    # check if df has less than 5 rows and 5 cols, skip
    if df.shape[0] < 5 or df.shape[1] < 5:
        return None
    
    # check if the df has more than 100 rows and cols, if so skip
    if df.shape[0] > 100 or df.shape[1] > 100:
        return None
    
    # if empty cells in the df is over 50%, skip
    if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) > 0.5:
        return None
    
    
    ## basic version of checking if the second to the last col is a pivot col
    key_cols = find_leftmost_key_cols(df)
    if key_cols > 0:
        # take the first key_cols cols as key, unpivot the rest
        key_cols_list = df.columns[:key_cols].tolist()
        # get the rest of the cols
        rest_cols_list = df.columns[key_cols:]
        
        df_unpivoted = df.melt(id_vars=key_cols_list, value_vars=rest_cols_list, var_name='variable', value_name='value')    
        return df_unpivoted
        
    return None
    
    

if __name__ == "__main__":

    # get the location of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = os.path.join(script_dir, "output", "pivot")
    os.makedirs(output_dir, exist_ok=True)
    
    
    # remove files in this output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # this will find all csv files in the data folder and its subfolders
    csv_files = glob.glob(os.path.join(script_dir, "data", "**", "*.csv"), recursive=True)
    for csv_file in csv_files:
        # get case_id from last subfolder name
        case_id = os.path.basename(os.path.dirname(csv_file))
        csv_file_name = os.path.basename(csv_file)
        
        if(case_id == '141490127' and csv_file_name == 'GL.csv'):
            debug = 1
        
        renamed_csv_file_name = f"{case_id}___{csv_file_name}"
        
        # if csv_file is empty, skip
        if os.path.getsize(csv_file) == 0:
            print(f"Skipping empty file: {csv_file}")
            continue
        try:
            df = pd.read_csv(csv_file, header=0)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        print(f"checking {csv_file}")
        
        re_pivot_df = re_pivot(df)
        # if reverse engineer pivot is successful, output the original df and the re_pivot_df 
        if re_pivot_df is not None:
            df.to_csv(os.path.join(output_dir, renamed_csv_file_name), index=False)
            re_pivot_df.to_csv(os.path.join(output_dir,  renamed_csv_file_name + "___re_pivotd.csv"), index=False)
