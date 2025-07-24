#!/bin/bash
#SBATCH --job-name=yolov5_train
#SBATCH --account=def-rsolisob
#SBATCH --time=0-01:00        
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=devel           
#SBATCH --output=logs/%x-%j.out  

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

# train the YOLOv5 model
python train_adv.py --img 640 --batch 16 --epochs 300 --data coco128.yaml --weights yolov5s.pt --cache
