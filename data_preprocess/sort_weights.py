import os
import shutil
from pathlib import Path

# 输入目录和输出目录
base_dir = Path("/fs-computility/plm/shared/zhuxuekai/reasoning_flow/verl/checkpoints/FlowRL/Qwen_32B/GRPO")
output_dir = Path("/fs-computility/plm/shared/zhuxuekai/reasoning_flow/verl/checkpoints/FlowRL/Qwen_32B/GRPO_merged")

# 创建输出目录
output_dir.mkdir(parents=True, exist_ok=True)

# 遍历所有 rank 目录
for rank_dir in base_dir.glob("rank*/grpo_baseline_qwen_32b_0630"):
    for step_dir in rank_dir.glob("global_step_*"):
        step_name = step_dir.name  # 如 global_step_200
        merged_step_dir = output_dir / step_name
        merged_step_dir.mkdir(parents=True, exist_ok=True)

        # 为避免文件名冲突，在复制时加上 rank 前缀
        rank_name = rank_dir.parts[-2]  # rank0, rank1...
        for item in step_dir.rglob("*"):
            if item.is_file():
                # 构造相对路径并加上 rank 前缀
                rel_path = item.relative_to(step_dir)
                target_path = merged_step_dir / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target_path)

print("✅ All steps merged across ranks.")
