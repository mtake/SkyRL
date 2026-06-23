set -x

BASENAME="$(basename "${BASH_SOURCE}" .sh)"
PROJECT_NAME_=${BASENAME}
PROJECT_NAME_=${PROJECT_NAME_#run_}
PROJECT_NAME_=${PROJECT_NAME_%_mtake}

# Colocated GRPO training+generation for Qwen3.5-0.8B on GSM8K.

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/models/run_qwen3.5_0.8b.sh

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${POLICY_MODEL:="Qwen/Qwen3.5-0.8B"}"
: "${NUM_GPUS:=4}"
: "${EPOCHS:=20}"
: "${LOGGER:=console}" # change to "console" to print to stdout
: "${LOG_PATH:="$HOME/tmp/skyrl-logs"}"

: "${INFERENCE_BACKEND:=vllm}"

: "${CKPTS_ROOT:="$HOME/ckpts"}"

# Qwen 3.5 flags
LANGUAGE_MODEL_ONLY=true
# there are issues with sample packing for GDN layers + flash attention
# https://github.com/huggingface/transformers/issues/44910 
# https://github.com/QwenLM/Qwen3.5/issues/104 
# disabling for now
REMOVE_MICROBATCH_PADDING=false # sample packing 

: "${PROJECT_NAME:="$PROJECT_NAME_"}"
: "${RUN_NAME:="qwen3.5-0.8b_fsdp"}"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$POLICY_MODEL \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.ref.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=$EPOCHS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.log_path="$LOG_PATH" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPTS_ROOT/qwen3.5-0.8b_ckpt" \
  $@
