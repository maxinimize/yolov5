#!/bin/bash
#SBATCH --job-name=yolov5_train
#SBATCH --account=def-rsolisob
#SBATCH --time=0-24:00        
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1           
#SBATCH --output=logs/%x-%j.out  

# Load necessary modules
module load python/3.10
module load gcc/9.3.0
module load cuda/11.4
module load opencv/4.11.0

# activate virtual environment
bash setup_env.sh
source yolov5_env/bin/activate

# train the YOLOv5 model
python train_adv.py --img 640 --batch 16 --epochs 300 --data coco128.yaml --weights yolov5s.pt --cache
