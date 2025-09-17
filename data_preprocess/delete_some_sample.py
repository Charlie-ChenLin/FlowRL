import pandas as pd

# 原文件路径
input_path = "outputs/gfn/llama_instruct/gfn_tempered_llama_Instruct_0624_global_step_50/test-output-16.parquet"
# 新文件保存路径（可以加个后缀，比如 `_trunc`）
output_path = "outputs/gfn/llama_instruct/gfn_tempered_llama_Instruct_0624_global_step_50/test-output-16_trunc.parquet"

# 读取 parquet 文件
df = pd.read_parquet(input_path)

# 删除最后5条数据
df_trimmed = df[:-10]

# 保存为新文件
df_trimmed.to_parquet(output_path, index=False)
