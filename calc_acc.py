import jsonlines
from utils import extract_answer_math
from tqdm import tqdm
from collections import defaultdict

if __name__ == "__main__":
    # with jsonlines.open("data/allenai/Llama-3.1-Tulu-3-8B-SFT/beam_search_completions_temp_0.8.jsonl") as f:
    # with jsonlines.open("data/allenai/Llama-3.1-Tulu-3-8B-SFT/MATH-test_beam_search_completions_temp_0.8.jsonl") as f:
    # with jsonlines.open("data/Qwen/Qwen2.5-Math-7B/beam_search_completions_range_None_to_None.jsonl") as f:
    # with jsonlines.open("data/Qwen/Qwen2.5-Math-7B/best_of_n_completions-temp_0.6-top_p_0.95-n_32-seed_42-iter_40-range_None_to_None.jsonl") as f:
    # with jsonlines.open("/p/llmresearch/thh9bk/search-and-learn/MATH-Qwen2.5-Math-7B-SFT_positive_advantage-bsz_1024-rollout_n_8-kl_coef_0.0-no_adv_normalization-classic_reward-vllm-0.8.2/global_step_112/actor/huggingface/best_of_n_completions-temp_0.6-top_p_0.95-n_256-seed_42-iter_40-range_None_to_None-subset_1000.jsonl") as f:
    with jsonlines.open("/p/llmresearch/thh9bk/search-and-learn/MATH-Qwen2.5-Math-7B-SFT_negative_advantage-bsz_1024-rollout_n_8-kl_coef_0.0-no_adv_normalization-classic_reward-vllm-0.8.2/global_step_112/actor/huggingface/best_of_n_completions-temp_0.6-top_p_0.95-n_256-seed_42-iter_40-range_None_to_None-subset_0.jsonl") as f:
    # with jsonlines.open("data/Qwen/Qwen2.5-Math-7B/beam_search_completions-temp_0.6-top_p_0.95-n_32-seed_42-iter_20-range_None_to_None.jsonl") as f:
        data = list(f)
    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    correct_count = defaultdict(int)
    accuracy = defaultdict(float)
    for k in k_values:
        correct_count[f"weighted_{k}"] = 0
        correct_count[f"maj_{k}"] = 0
        correct_count[f"naive_{k}"] = 0

    for ex in tqdm(data):
        gold_answer = extract_answer_math(ex["solution"])
        for k in k_values:
            correct_count[f"weighted_{k}"] += (extract_answer_math(ex[f"pred_weighted@{k}"]) == gold_answer)
            correct_count[f"maj_{k}"] += (extract_answer_math(ex[f"pred_maj@{k}"]) == gold_answer)
            correct_count[f"naive_{k}"] += (extract_answer_math(ex[f"pred_naive@{k}"]) == gold_answer)

    for k in k_values:
        accuracy[f"maj_{k}"] = correct_count[f"maj_{k}"] / len(data)
        accuracy[f"naive_{k}"] = correct_count[f"naive_{k}"] / len(data)
        accuracy[f"weighted_{k}"] = correct_count[f"weighted_{k}"] / len(data)

    for k in k_values:
        print(f"acc_maj_{k}: {accuracy[f'maj_{k}']}, acc_naive_{k}: {accuracy[f'naive_{k}']}, acc_weighted_{k}: {accuracy[f'weighted_{k}']}")
