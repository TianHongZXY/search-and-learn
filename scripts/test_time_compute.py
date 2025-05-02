#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    if os.path.exists(f"{config.output_dir}/{config.approach}_completions_range_{config.start_id}_to_{config.end_id}.jsonl"):
        logger.info(f"Output file {config.approach}_completions_range_{config.start_id}_to_{config.end_id}.jsonl already exists.")
        logger.info("Passed because of existing file!")
        return

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )
    prm = load_prm(config)

    dataset = get_dataset(config)
    if config.start_id is not None and config.end_id is not None:
        dataset = dataset.select(range(config.start_id, config.end_id))
    for i in range(0, len(dataset), config.save_batch_size):
        batch = dataset.select(range(i, min(i + config.save_batch_size, len(dataset))))
        batch = batch.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "prm": prm},
            desc="Running search",
            load_from_cache_file=False,
        )
    # dataset = dataset.map(
    #     approach_fn,
    #     batched=True,
    #     batch_size=config.search_batch_size,
    #     fn_kwargs={"config": config, "llm": llm, "prm": prm},
    #     desc="Running search",
    #     load_from_cache_file=False,
    # )

        batch = score(batch, config)

        save_dataset(batch, config, i)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
