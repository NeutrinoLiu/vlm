#!/usr/bin/env python3
import os
import subprocess
import shutil
import random
import datetime
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=".", config_name="default")
def main(cfg: DictConfig):
    # Generate timestamp
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set run_name and output_dir if not provided
    if not cfg.output_dir:
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    
    ts = cfg.output_dir.split("/")[-1]

    if not cfg.run_name:
        cfg.run_name = f"qwen2vl-3b-lora-{ts}"

    # Set random port if not provided
    if cfg.master_port == -1:
        cfg.master_port = random.randint(20001, 29999)
    
    if cfg.nproc_per_node == -1:
        import torch
        cfg.nproc_per_node = torch.cuda.device_count()
        print(f"Number of available GPUs: {cfg.nproc_per_node}")


    # Build command-line arguments for the training script

    # cache_map = {
    #     "Qwen/Qwen2.5-VL-3B-Instruct": cfg.cache_3b,
    #     "Qwen/Qwen2.5-VL-7B-Instruct": cfg.cache_7b,
    #     "Qwen/Qwen2.5-VL-32B-Instruct": cfg.cache_32b,
    # }
    # assert cfg.llm in cache_map, f"using {cfg.llm} cache path {cfg.cache_path}"

    # cache_path = cache_map.get(cfg.llm, cfg.cache_path)

    args = [
        "--lora_llm", str(cfg.lora_llm),
        "--lora_r", str(cfg.lora_r),
        "--lora_alpha", str(cfg.lora_alpha),
        "--deepspeed", cfg.deepspeed,
        "--model_name_or_path", cfg.llm,
        # "--model_cache_path", cache_path,
        "--dataset_use", cfg.datasets,
        "--data_flatten", str(cfg.data_flatten),
        "--data_packing", str(cfg.data_packing),
        "--tune_mm_vision", str(cfg.tune_mm_vision),
        "--tune_mm_mlp", str(cfg.tune_mm_mlp),
        "--tune_mm_llm", str(cfg.tune_mm_llm),
    ]
    
    if cfg.bf16:
        args.append("--bf16")
    
    args.extend([
        "--output_dir", cfg.output_dir,
        "--num_train_epochs", str(cfg.num_train_epochs),
        "--per_device_train_batch_size", str(cfg.batch_size),
        "--per_device_eval_batch_size", str(cfg.batch_size * 2),
        "--gradient_accumulation_steps", str(cfg.grad_accum_steps),
        # "--max_pixels", str(cfg.max_pixels),
        # "--min_pixels", str(cfg.min_pixels),
        "--eval_strategy", cfg.eval_strategy,
        "--save_strategy", cfg.save_strategy,
        "--save_steps", str(cfg.save_steps),
        "--save_total_limit", str(cfg.save_total_limit),
        "--learning_rate", str(cfg.lr),
        "--weight_decay", str(cfg.weight_decay),
        "--warmup_ratio", str(cfg.warmup_ratio),
        "--max_grad_norm", str(cfg.max_grad_norm),
        "--lr_scheduler_type", cfg.lr_scheduler_type,
        "--logging_steps", str(cfg.logging_steps),
        "--model_max_length", str(cfg.model_max_length),
        "--gradient_checkpointing", str(cfg.gradient_checkpointing),
        "--dataloader_num_workers", str(cfg.dataloader_num_workers),
        "--run_name", cfg.run_name,
        "--report_to", cfg.report_to
    ])
    
    # Prepare torchrun command
    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={cfg.nproc_per_node}",
        f"--master_addr={cfg.master_addr}",
        f"--master_port={cfg.master_port}",
        cfg.entry_file
    ]
    
    # Execute the command
    full_command = torchrun_cmd + args
    print("Executing command:")
    print(" ".join(full_command))
    subprocess.run(full_command)

if __name__ == "__main__":
    main()