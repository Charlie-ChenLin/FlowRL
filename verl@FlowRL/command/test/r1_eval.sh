
MODEL_NAME=gfn_important_sampling_15_step200_0606
OUTPUT_PATH=/fs-computility/plm/shared/zhuxuekai/reasoning_flow/outputs/$MODEL_NAME

# Evaluation
python3 -m recipe.r1.main_eval \
    data.path=$OUTPUT_PATH/test-output-16.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=recipe/r1/reward_score.py \
    custom_reward_function.name=reward_func