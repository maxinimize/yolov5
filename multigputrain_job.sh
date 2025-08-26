#!/bin/bash
#SBATCH --job-name=yolov5_train_ddp
#SBATCH --account=def-rsolisob
#SBATCH --time=0-12:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --output=logs/%x-%j.out
# SBATCH --qos=devel

set -euo pipefail

# === Modules & venv ===
module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load opencv/4.11.0

source yolov5_env/bin/activate

# OpenCV for cv2
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH

# BLAS threading guard
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# PyTorch CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# NCCL
export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_DEBUG=WARN

export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=1

# ==== Determine number of processes per node (GPUs) & dataloader workers ====
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}
else
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || true)
  NPROC_PER_NODE=${NGPU:-1}
fi

THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC_PER_NODE ))
if [ "$THREADS_PER_PROC" -lt 1 ]; then THREADS_PER_PROC=1; fi
export OMP_NUM_THREADS=$THREADS_PER_PROC
export OMP_THREAD_LIMIT=$THREADS_PER_PROC

NUM_WORKERS=$(( THREADS_PER_PROC - 1 ))
if [ "$NUM_WORKERS" -lt 1 ]; then NUM_WORKERS=1; fi

# Global batch size (must be divisible by world_size)
GLOBAL_BATCH=24

# === Debugging info ===
echo "===== debug env ====="
echo "Host: $(hostname)"
echo "GPUs on node: ${SLURM_GPUS_ON_NODE:-<unset>}"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "OMP_NUM_THREADS (per-proc): $OMP_NUM_THREADS"
echo "DataLoader workers (per-proc): $NUM_WORKERS"
echo "====================="

LOGDIR=logs/ddp_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

# === Training (DDP) ===
torchrun --standalone --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} \
  --tee 3 --log-dir "$LOGDIR" \
  train_adv.py \
    --img 640 \
    --batch-size ${GLOBAL_BATCH} \
    --epochs 5 \
    --data coco.yaml \
    --weights yolov5x.pt \
    --attack-weights yolov5x.pt \
    --cache ram \
    --workers ${NUM_WORKERS} \
    --patience 500