#!/bin/bash

# python virtualenv
source .env/bin/activate

# distributed settings
JOB_ID=$(date +%s%N)
export MASTER_ADDR=10.130.184.10
export MASTER_PORT=12802

echo "MASTER_ADDR=${MASTER_ADDR}"

NODE_TYPE="a100"
export NUM_GPU_PER_NODE=8

mkdir -p ./hostfile

SGE_JOB_HOSTLIST=scripts/mdx/hostfile
export SGE_JOB_HOSTLIST

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

NUM_NODES=16
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# model config
# llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008 # intermediate size (HuggingFace)
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=2   # fixed
PIPELINE_PARALLEL_SIZE=2 # num layers 32: Llama-2 7B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
# 今回は約100B Tokensなので 1/10

LR=1e-4
MIN_LR=3.3e-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/model/fujii/jalm-tokenizer-private/tokenizer/jalm_llama_okazaki_lab_cc_nfkc_16k_aligned_8/merged_tokenizer_sp/jalm_llama.model
CHECKPOINT_DIR=/model/fujii/checkpoints/Llama-2-7b/
CHECKPOINT_SAVE_DIR=/model/fujii/checkpoints/Llama-2-7b/

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/model/fujii/datasets/pretraining/llama

DATA_PATH=""

DATA_PATH="${DATA_PATH} 1672543873 ${DATASET_DIR}/ja_wiki_merged_train_text_document"


# job name
JOB_NAME="base-line-1F1B-llama-2-7b-${NUM_NODES}node"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
  -bind-to none -map-by slot \
  -x PATH \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --save-interval 5000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --use-rotary-position-embeddings \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --no-position-embedding \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "zero-bubble-pipeline" \
  --wandb-entity "llm-jp"
