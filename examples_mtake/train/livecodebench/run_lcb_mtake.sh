set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-3B-Instruct on SearchR1 data.
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/livecodebench/run_lcb.sh

: "${DATA_DIR:="$HOME/data/lcb"}"
: "${POLICY_MODEL:="Qwen/Qwen2.5-3B-Instruct"}"
: "${NUM_GPUS:=8}"
: "${EPOCHS:=1}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout
: "${LOG_PATH:="$HOME/tmp/skyrl-logs"}"

: "${INFERENCE_BACKEND:=vllm}"

: "${CKPTS_ROOT:="$HOME/ckpts"}"

# NOTE (sumanthrh): micro_train_batch_size and micro_forward_batch_size can be tuned
uv run --isolated --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data="['${DATA_DIR}/deepcoder_train.json']" \
  data.val_data="['${DATA_DIR}/test_livecodebench.json']" \
  trainer.policy.model.path="$POLICY_MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.train_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=29000 \
  generator.max_input_length=29000 \
  generator.sampling_params.max_generate_length=3000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.ckpt_interval=100000 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  environment.env_class=lcb \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  trainer.logger="$LOGGER" \
  trainer.log_path="$LOG_PATH" \
  trainer.project_name="skyrl" \
  trainer.run_name="skyrlcode_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPTS_ROOT/lcb_3B_ckpt" \
  trainer.epochs=$EPOCHS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@
