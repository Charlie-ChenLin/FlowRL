
MODEL_NAME=grpo_qwen7b_tr-dapo_0602_step300_merged
MODEL_PATH=/fs-computility/plm/shared/zhuxuekai/reasoning_flow/results_model/merged_model/$MODEL_NAME
DATA_PATH=/fs-computility/plm/shared/zhuxuekai/reasoning_flow/data/math_test
OUTPUT_PATH=/fs-computility/plm/shared/zhuxuekai/reasoning_flow/outputs/$MODEL_NAME

n_gpus_per_node=2

# Generation
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=16 \
    data.output_path=$OUTPUT_PATH/test-output-16.parquet \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.7 \
    rollout.max_num_batched_tokens=65536

# Evaluation
# python3 -m recipe.r1.main_eval \
#     data.path=$OUTPUT_PATH/test-output-8.parquet \
#     data.prompt_key=prompt \
#     data.response_key=responses \
#     custom_reward_function.path=recipe/r1/reward_score.py \
#     custom_reward_function.name=reward_func