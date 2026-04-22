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

#     # Function to check if a column has a repeating pattern
# def has_repeating_pattern(df, col):
#     values = df[col].values
#     n = len(values)
    
#     # get distinct vals in col
#     unique_values = pd.unique(values)
#     if len(unique_values) < 3:
#         return False, None
    
#     len_pattern = len(unique_values)
#     # repeat the first len_pattern values to match the length of the column

#     pattern = values[:len_pattern]  # Consider the first 'length' elements as a pattern
#     repeated = pattern * (n // len_pattern)
#     if np.array_equal(values, repeated): # Check if the repeated pattern matches the column
#         return True, pattern

#     return False, None  # No repeating pattern found



    # check if the second to the last col is the pivot col
def has_repeating_pattern_simple(df):
    # get the second to the last col
    col = df.columns[-2]
    # check if the col has less than 5 unique vals, skip
    if df[col].nunique() > 50 or df[col].nunique() < 3:
        return False
    # check if the col is numeric, skip
    if pd.api.types.is_numeric_dtype(df[col]):
        return False
    
    # check how many unique vals in the col,
    unique_pivot_col_vals = df[col].unique()
    # count how many unique val-combination in all cols to the left of col
    index_cols = df.columns[:-2]
    unique_index_cols_vals = df[index_cols].drop_duplicates().shape[0]

    # can pivot if # of rows check out
    if(df.shape[0] == unique_index_cols_vals * len(unique_pivot_col_vals)):
        return True
    
    return False
    

# check if a df is suitable to reverse-engineer using unpivot
def re_unpivot(df):
    # check if df has less than 5 rows and 5 cols, skip
    if df.shape[0] < 5 or df.shape[1] < 5:
        return None
    
    # check if the df has more than 100 rows and cols, if so skip
    # if df.shape[0] > 100 or df.shape[1] > 100:
    #     return None
    
    # if empty cells in the df is over 50%, skip
    if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) > 0.5:
        return None
    
    
    ## basic version of checking if the second to the last col is a pivot col
    if has_repeating_pattern_simple(df):
        pivot_col = df.columns[-2]
        value_col = df.columns[-1]
        index_cols = df.columns[:-2]
        print(df.dtypes)
        print(df.columns)

        # now pivot
        df_pivoted = df.pivot(index=index_cols, columns=pivot_col, values=value_col).reset_index()
        return df_pivoted
    
    return None
    
    
    # # iterate all cols, find suitable col that can pivot (with perfectly repeating vals)
    # for col in df.columns:
    #     # check if the col has less than 5 unique vals, skip
    #     if df[col].nunique() > 50:
    #         continue
    #     # check if the col is numeric, skip
    #     if pd.api.types.is_numeric_dtype(df[col]):
    #         continue
    
    #     # now try to pivot the col, and see if its vals are perfectly repeating, if so can pivot without info loss
    #     ## check if the col has repeating valsn
    #     has_pattern, pattern = has_repeating_pattern(df, col)
    #     if has_pattern:
    #         # pivot the df using this col
    #         ## if we do not use aggressive pivot, we only pivot if the col is the second to the last col, and we treat the last as vals
    #         if (use_aggressive_pivot == False):
    #             # see if col is the second to the last col in df
    #             if col != df.columns[-2]:
    #                 continue
    #             # if col is the second to the last col, we use the last col as the value col
    #             value_col = df.columns[-1]
    #             # use all cols to the left of col, as index cols
    #             index_cols = df.columns[:-2]
    #             # now pivot
    #             df_pivoted = df.pivot_table(index=index_cols, columns=col, values=value_col, aggfunc='first').reset_index()
                
    #             return df_pivoted
    #         ## if we do aggressive pivot, we may only use a subset of cols as index cols, which may drop some cols from the orig df
    #         else:
    #             # raise exception that this is not implemented yet
    #             raise NotImplementedError("Aggressive pivot is not implemented yet.")
    #             pass
    
    # return None
    

if __name__ == "__main__":

    # get the location of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = os.path.join(script_dir, "output", "unpivot")
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
        
        
        re_unpivot_df = re_unpivot(df)
        # if reverse engineer unpivot is successful, output the original df and the re_unpivot_df 
        if re_unpivot_df is not None:
            df.to_csv(os.path.join(output_dir, renamed_csv_file_name), index=False)
            re_unpivot_df.to_csv(os.path.join(output_dir,  renamed_csv_file_name + "___re_unpivotd.csv"), index=False)
