#!/bin/bash
#SBATCH --job-name=yolov5_val
#SBATCH --account=def-rsolisob
#SBATCH --time=0-1:00        
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

# train the YOLOv5 model
python val_adv.py --weights runs/train/exp3/weights/best.pt --attack-weights yolov5x.pt --data coco128.yaml --img 640 --half
# python val_adv.py --weights yolov5x.pt --attack-weights yolov5x.pt --data coco128.yaml --img 640 --half