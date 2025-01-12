#!/bin/bash
# ====================================================
#   Copyright (C) 2025  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : thh9bk@virginia.edu
#   File Name     : run.sh
#   Last Modified : 2025-01-10 12:20
#   Describe      : 
#
# ====================================================



# python scripts/test_time_compute.py recipes/Tulu-3-8B-SFT/beam_search.yaml --dataset_name=lighteval/MATH --dataset_split=train --custom_chat_template=none --num_iterations=20 --start_id=700 --end_id=900 --push_to_hub=true --hub_dataset_id=TianHongZXY/Tulu-3-8B-SFT-beam_search-completions-temp_0.8-range_700_to_900; python scripts/test_time_compute.py recipes/Tulu-3-8B-SFT/beam_search.yaml --dataset_name=lighteval/MATH --dataset_split=train --custom_chat_template=none --num_iterations=20 --start_id=900 --end_id=1200 --push_to_hub=true --hub_dataset_id=TianHongZXY/Tulu-3-8B-SFT-beam_search-completions-temp_0.8-range_900_to_1200
# export CUDA_LAUNCH_BLOCKING=1
for start_id in $(seq 1900 20 6980); do
  end_id=$((start_id + 20))
  python scripts/test_time_compute.py recipes/Tulu-3-8B-SFT/beam_search.yaml \
    --dataset_name=lighteval/MATH \
    --dataset_split=train \
    --custom_chat_template=none \
    --num_iterations=20 \
    --start_id=${start_id} \
    --end_id=${end_id} \
    # --push_to_hub=true \
    # --hub_dataset_id=TianHongZXY/Tulu-3-8B-SFT-beam_search-completions-temp_0.8-range_${start_id}_to_${end_id}
  sleep 60
done
