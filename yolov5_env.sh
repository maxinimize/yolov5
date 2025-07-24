#!/bin/bash

# env name
VENV_PATH="yolov5_env"

# load necessary modules on SHARCNET
module load python/3.10
module load gcc/13.3
module load cuda/11.8.0
module load opencv/4.11.0

# Check if the virtual environment already exists
if [ -d "$VENV_PATH" ]; then
    echo "A virtual environment already exists, activating..."
    source $VENV_PATH/bin/activate
    echo "Activated virtual environment: $VENV_PATH"
    exit 1
else
    echo "A virtual environment does not exist, creating a new one..."
    python -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    pip install --upgrade pip
    
    # Explicitly install torch for CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    if [ -f "requirements_sharcnet.txt" ]; then
        pip install -r requirements_sharcnet.txt
    elif [ -f "requirements.txt" ]; then
        # # Create a temporary requirements file, excluding opencv-
        grep -v "opencv" requirements.txt > temp_requirements.txt
        pip install -r temp_requirements.txt
        rm temp_requirements.txt
    else
        echo "requirements.txt not found, Exiting..."
        exit 1
    fi
    
    echo "Installed YOLOv5 dependencies from requirements file."
fi