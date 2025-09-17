from datasets import load_dataset
import pickle
import json

# data_path = "data/math"
# data_path = "data/gsm8k"
data_path = "data/game24"

train_data = load_dataset('parquet', data_files='{}/train.parquet'.format(data_path))['train']

# def filter_label(example):
#     # try:
#     example['extra_info'])  
#     return extra_info.get('label') == 1  
#     # except json.JSONDecodeError:
#         # return False  

filtered_data = train_data.filter(lambda x: x["extra_info"]["label"] == 1)

save_data_path = "data/game24-sft"
filtered_data.to_parquet(f'{save_data_path}/train.parquet')
