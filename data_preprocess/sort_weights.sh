#!/bin/bash

# 设置路径
BASE_DIR="/fs-computility/plm/shared/zhuxuekai/reasoning_flow/verl/checkpoints/FlowRL/Qwen_32B/RPP"
OUTPUT_DIR="/fs-computility/plm/shared/zhuxuekai/reasoning_flow/verl/checkpoints/FlowRL/Qwen_32B/RPP_merged"

# 遍历每个 rank
for RANK_PATH in "$BASE_DIR"/rank*/rpp_32b_qwen_0701; do
  # 遍历每个 global_step_xxx
  for STEP_PATH in "$RANK_PATH"/global_step_*; do
    STEP_NAME=$(basename "$STEP_PATH")
    DEST_DIR="$OUTPUT_DIR/$STEP_NAME"
    mkdir -p "$DEST_DIR"
    # 用 cp 递归复制内容（不保留时间戳，不去重）
    cp -r "$STEP_PATH"/* "$DEST_DIR"/ 2>/dev/null
  done
done

echo "✅ Merge finished using cp."
