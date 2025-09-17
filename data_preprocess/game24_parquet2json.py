from datasets import load_dataset
import pickle
import json
import os
# data_path = "data/math"
# data_path = "data/gsm8k"


data_path = "data/game24"

train_data = load_dataset('parquet', data_files='{}/train.parquet'.format(data_path))['train']
processed_data = []
for iterm in train_data:
    if iterm["extra_info"]["label"] == 1:
        iterm_dict = { "instruction": {},
                        "input": iterm["prompt"][0]["content"],
                        "output": iterm["generate_data"],}
        processed_data.append(json.dumps(iterm_dict))

save_data_path = "data/game24-sft"
os.makedirs(save_data_path, exist_ok=True)
with open(f"{save_data_path}/train.json", "w") as f_in:
    for i in processed_data:
        f_in.write(i+"\n")