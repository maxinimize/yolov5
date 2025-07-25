#!/bin/bash
#SBATCH --job-name=yolov5setup
#SBATCH --account=def-rsolisob
#SBATCH --time=0-00:15
#SBATCH --mem=1G
# env name
VENV_PATH="yolov5_env"

# load necessary modules on SHARCNET
module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load opencv/4.11.0

# Check if the virtual environment already exists
if [ -d "$VENV_PATH" ]; then
    echo "A virtual environment already exists, activating..."
    source $VENV_PATH/bin/activate
    echo "Activated virtual environment: $VENV_PATH"
    exit 1
else
    echo "A virtual environment does not exist, creating a new one..."
    virtualenv --no-download $VENV_PATH
    source $VENV_PATH/bin/activate
    pip install --upgrade pip

    # Explicitly install torch 
    # should be for CUDA 12
    pip install --no-index torch torchvision torchtext torchaudio

    if [ -f "requirements_sharcnet.txt" ]; then
        pip install --no-index -r requirements_sharcnet.txt
    elif [ -f "requirements.txt" ]; then
        # # Create a temporary requirements file, excluding opencv-
        grep -v "opencv" requirements.txt > temp_requirements.txt
        pip install --no-index -r temp_requirements.txt
        pip install --no-index ultralytics --no-deps
        # set PYTHONPATH
        export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH
        rm temp_requirements.txt
    else
        echo "requirements.txt not found, Exiting..."
        exit 1
    fi
    
    echo "Installed YOLOv5 dependencies from requirements file."
fi
