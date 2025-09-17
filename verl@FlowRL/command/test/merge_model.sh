#!/bin/bash
set -x 

ROOT_DIR="/fs-computility/plm/shared/zhuxuekai/reasoning_flow/"

BACKEND="fsdp"  
LOCAL_DIR=$ROOT_DIR"verl/checkpoints/FlowRL/grpo_llama_baseline_qwen7b_tr-dapo_0602/global_step_200/actor"
TARGET_DIR=$ROOT_DIR"merged_model/grpo_llama_step200"

PYTHONPATH=. python scripts/model_merger.py merge \
  --backend $BACKEND \
  --local_dir $LOCAL_DIR \
  --target_dir $TARGET_DIR

