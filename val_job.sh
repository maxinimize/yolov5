#!/bin/bash
#SBATCH --job-name=yolov5_val
#SBATCH --account=def-rsolisob
#SBATCH --time=0-12:00        
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out  

# Load necessary modules SBATCH --qos=devel
module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load opencv/4.11.0

# activate virtual environment
source yolov5_env/bin/activate

# set OpenCV path for cv2
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# train the YOLOv5 model
python val_adv.py --weights runs/train/exp25/weights/best.pt --attack-weights yolov5l.pt --data coco_val.yaml --img 640 --half
# python val_adv.py --weights yolov5l.pt --attack-weights yolov5l.pt --data coco_val.yaml --img 640 --half
# python val.py --weights runs/train/exp25/weights/best.pt --data coco_val.yaml --img 640 --half
# python val.py --weights yolov5l.pt --data coco_val.yaml --img 640 --half
