#!/bin/bash
#SBATCH --job-name=yolov5_train
#SBATCH --account=def-rsolisob
#SBATCH --time=0-12:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x-%j.out  
# SBATCH --qos=devel

set -euo pipefail

# Load necessary modules
module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load opencv/4.11.0

# activate virtual environment
source yolov5_env/bin/activate

# set OpenCV path for cv2
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH

# Fix: prevent BLAS thread explosion (for FlexiBLAS + BLIS)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional: force FlexiBLAS to use openblas
# export FLEXIBLAS_BACKEND=openblas

# train the YOLOv5 model
# workers correspond to cpu cores used, default is 8 but now its explicit. More means training goes faster.
python train_adv.py \
  --img 640 --batch 32 --epochs 50 \
  --data coco.yaml \
  --weights yolov5x.pt \
  --attack-weights yolov5x.pt \
  --device 0 \
  --cache ram \
  --workers 20 \
  --patience 500