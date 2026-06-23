set -x

BASENAME="$(basename "${BASH_SOURCE}" .sh)"
PROJECT_NAME_=${BASENAME}
PROJECT_NAME_=${PROJECT_NAME_#run_}
PROJECT_NAME_=${PROJECT_NAME_%_mtake}

# Colocated GRPO training+generation for Qwen1.5-MoE-A2.7B-Chat on GSM8K.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${POLICY_MODEL:="Qwen/Qwen1.5-MoE-A2.7B-Chat"}"
: "${NUM_GPUS:=4}"
: "${EPOCHS:=20}"
: "${LOGGER:=console}" # change to "console" to print to stdout
: "${LOG_PATH:="$HOME/tmp/skyrl-logs"}"

: "${INFERENCE_BACKEND:=vllm}"

: "${CKPTS_ROOT:="$HOME/ckpts"}"

TP_SIZE=$NUM_GPUS
EP_SIZE=$NUM_GPUS
DP_SIZE=1

: "${PROJECT_NAME:="$PROJECT_NAME_"}"
: "${RUN_NAME:="moe_test"}"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$POLICY_MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.expert_parallel_size=$EP_SIZE \
  generator.inference_engine.data_parallel_size=$DP_SIZE \
  trainer.epochs=$EPOCHS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
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
  trainer.ckpt_path="$CKPTS_ROOT/moe_qwen_a2_7b_ckpt" \
  $@
