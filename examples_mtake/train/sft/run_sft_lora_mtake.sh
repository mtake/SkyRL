#!/bin/bash
set -x

# SFT LoRA training with FSDP backend for Qwen2.5-0.5B-Instruct
#
# This script runs supervised fine-tuning with LoRA adapters using FSDP on
# 1 GPU with the Alpaca dataset. Only a small fraction of parameters are
# trainable (controlled by lora.rank / lora.alpha), making this suitable
# for memory-constrained setups.
#
# Usage:
#   bash examples/train/sft/run_sft_lora.sh [extra overrides...]
#
# Example:
#   bash examples/train/sft/run_sft_lora.sh num_steps=20 model.lora.rank=64

: "${MODEL:="Qwen/Qwen2.5-0.5B-Instruct"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=console}" # change to "console" to print to stdout
: "${LOG_PATH:="$HOME/tmp/skyrl-logs"}"

: "${CKPTS_ROOT:="$HOME/ckpts"}"

: "${PROJECT_NAME:="skyrl_sft_lora"}"
: "${RUN_NAME:="skyrl_sft_lora_fsdp_run"}"

uv run --isolated --extra fsdp \
    python -m skyrl.train.main_sft \
    strategy=fsdp \
    model.path="$MODEL" \
    model.lora.rank=32 \
    model.lora.alpha=16 \
    model.lora.target_modules=all-linear \
    dataset_name=yahma/alpaca-cleaned \
    dataset_split="train[:100]" \
    messages_key=messages \
    max_length=512 \
    num_steps=10 \
    batch_size=4 \
    micro_train_batch_size_per_gpu=2 \
    remove_microbatch_padding=true \
    seed=42 \
    optimizer_config.lr=2e-5 \
    optimizer_config.weight_decay=0.0 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=0 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=$NUM_GPUS \
    fsdp_config.cpu_offload=false \
    fsdp_config.reshard_after_forward=true \
    logger="$LOGGER" \
    log_path="$LOG_PATH" \
    project_name="$PROJECT_NAME" \
    run_name="$RUN_NAME" \
    ckpt_path="$CKPTS_ROOT/sft_lora_ckpt" \
    ckpt_interval=0 \
    resume_from="" \
    "$@"
