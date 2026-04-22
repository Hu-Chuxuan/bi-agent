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



""" This function transposes a DataFrame with the first row as header, into a transposed DF.
It first demotes the current column headers as the first data row, sets the first column (actual col header) as the row-index, and then tranposes the DataFrame.
"""
def helper_transpose_df_with_header(df):
    # Step 1: Demote current column headers to first row
    header_row = pd.DataFrame([df.columns], columns=df.columns)
    df_with_header_row = pd.concat([header_row, df], ignore_index=True)
    df_with_header_row.columns = range(df_with_header_row.shape[1])  # Reset col headers to 0,1,2,...
    
    # Step 2: Make first column the row index
    df_with_header_row = df_with_header_row.set_index(0)
    df_with_header_row.index.name = None

    # Step 3: Transpose
    df_transposed = df_with_header_row.T.reset_index(drop=True)
    
    return df_transposed
    

# check if a df is suitable to reverse-engineer using transpose
def re_transpose(df):
    # check if df has less than 5 rows and 5 cols, skip
    if df.shape[0] < 5 or df.shape[1] < 5:
        return None
    # check if the first col of df are all unique vals
    if not df.iloc[:, 0].is_unique:
        return None
    # if all vals in the first col are either integers or floating point numbers, skip
    if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        return None
    # if empty cells in the df is over 50%, skip
    if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) > 0.5:
        return None
    # check if the df has more than 100 rows and cols, if so skip
    if df.shape[0] > 100 or df.shape[1] > 100:
        return None
    
    return helper_transpose_df_with_header(df)

if __name__ == "__main__":

    # get the location of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = os.path.join(script_dir, "output", "transpose")
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
        
        renamed_csv_file_name = f"{case_id}___{csv_file_name}"
        
        # if csv_file is empty, skip
        if os.path.getsize(csv_file) == 0:
            print(f"Skipping empty file: {csv_file}")
            continue
        df = pd.read_csv(csv_file, header=0)
        print(f"checking {csv_file}")
        
        
        re_transpose_df = re_transpose(df)
        # if reverse engineer transpose is successful, output the original df and the re_transpose_df 
        if re_transpose_df is not None:
            df.to_csv(os.path.join(output_dir, renamed_csv_file_name), index=False)
            re_transpose_df.to_csv(os.path.join(output_dir,  renamed_csv_file_name + "___re_transposed.csv"), index=False)
