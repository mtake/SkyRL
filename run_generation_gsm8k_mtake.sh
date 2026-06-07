#!/usr/bin/env bash

#
# OK with 1 GPU
#

# for macOS
if command -v gdate &> /dev/null
then
    DATE_CMD=gdate
else
    DATE_CMD=date
fi

START_TIME="$(${DATE_CMD} +%s)"
START_TIME_STR="$(${DATE_CMD} -d @${START_TIME} +%Y%m%d-%H%M%S)"
BASENAME="$(basename "${BASH_SOURCE}" .sh)"
HOSTNAME_S="$(hostname -s)"
LOGFILE="${BASENAME}-${START_TIME_STR}-${HOSTNAME_S}.log"
echo "XXX LOGFILE ${LOGFILE}" | tee -a ${LOGFILE}
echo "XXX DATETIME ${START_TIME_STR}" | tee -a ${LOGFILE}

# count gpus
if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=0
fi
echo "XXX NUM_GPUS: ${NUM_GPUS}" | tee -a ${LOGFILE}

if (( NUM_GPUS == 0 )); then
    echo "ERROR: A GPU is required to run this command. Exiting..." | tee -a ${LOGFILE}
    exit 1
fi

#VENV=../../.venv
VENV=.venv
if [[ -d "${VENV}" ]]; then
    source "${VENV}/bin/activate"
fi

# @@@ahoaho XXX
# NOTE start Ray if not running.
unset _RAY_STARTED
if ! ray status > /dev/null 2>&1; then
    echo "XXX Starting Ray..."
    ray start --head
    _RAY_STARTED=1
fi

ENV=""
ENV="TOKENIZERS_PARALLELISM=false ${ENV}"
ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${ENV}"  # deprecated
ENV="PYTORCH_ALLOC_CONF=expandable_segments:True ${ENV}"
ENV="NCCL_DEBUG=INFO ${ENV}"

if true; then
ENV="CUDA_LAUNCH_BLOCKING=1 ${ENV}"
ENV="TORCH_USE_CUDA_DSA=1 ${ENV}"
fi

if false; then
ENV="TORCH_CPP_LOG_LEVEL=INFO ${ENV}"
ENV="TORCH_DISTRIBUTED_DEBUG=DETAIL ${ENV}"

ENV="NCCL_DEBUG_SUBSYS=ALL ${ENV}"

ENV="NCCL_ASYNC_ERROR_HANDLING=1 ${ENV}"  # deprecated

ENV="TORCH_NCCL_ASYNC_ERROR_HANDLING=1 ${ENV}"

#ENV="NCCL_P2P_DISABLE=1 ${ENV}"
#ENV="NCCL_SHM_DISABLE=1 ${ENV}"
#ENV="NCCL_IB_DISABLE=1 ${ENV}"
fi

ENV="DATA_DIR=${HOME}/data/gsm8k ${ENV}"
ENV="NUM_GPUS=${NUM_GPUS} ${ENV}"
ENV="LOGGER=console ${ENV}"
ENV="INFERENCE_BACKEND=vllm ${ENV}"

# ENV="CKPTS_ROOT=${HOME}/ckpts ${ENV}"
ENV="CKPTS_ROOT=${PWD}/ckpts ${ENV}"

echo "================== ENVIRONMENT VARIABLES ===================" | tee -a ${LOGFILE}
env 2>&1 | tee -a ${LOGFILE}
echo "============================================================" | tee -a ${LOGFILE}

# @@@ahoaho XXX
#cmd="${ENV}bash examples/train/gsm8k/run_generation_gsm8k.sh"
cmd="${ENV}bash examples/train/gsm8k/run_generation_gsm8k_mtake.sh"
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

# @@@ahoaho XXX
if [[ -n "${_RAY_STARTED}" ]]; then
    echo "XXX Stopping Ray..."
    ray stop
fi

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
