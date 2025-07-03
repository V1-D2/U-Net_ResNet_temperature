#!/bin/bash
#SBATCH --job-name=amsr2_8x_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=48:00:00

echo "============================================"
echo "AMSR2 8x Super-Resolution Training Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 80GB per GPU"
echo "============================================"

# Set environment variables for memory management
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Change to project directory
cd /home/vdidur/U-Net_ResNet_temperature

# Check/Install packages
echo "Checking required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user --quiet torch torchvision numpy matplotlib scikit-learn tqdm psutil

# Test GPU environment
echo "Testing GPU environment..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    # Clear any existing allocations
    torch.cuda.empty_cache()
    print('GPU cache cleared')
"

# Create necessary directories
mkdir -p ./models
mkdir -p ./logs
mkdir -p ./results

echo "============================================"
echo "Starting 8x Super-Resolution Training..."

# Run the memory-safe GPU sequential trainer
# IMPORTANT: Use small batch size and gradient accumulation for memory safety
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/U-Net_ResNet_temperature:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python gpu_sequential_amsr2_fixed.py \
    --npz-dir /home/vdidur/temperature_sr_project/data \
    --max-files 4 \
    --epochs-per-file 2 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --lr 1e-4 \
    --use-amp \
    --target-height 2000 \
    --target-width 200 \
    --save-path ./models/best_amsr2_8x_model.pth

echo "============================================"
echo "Training Finished: $(date)"
echo "============================================"

# Show results
echo "Results saved in:"
ls -la ./models/best_amsr2_8x_model.pth 2>/dev/null || echo "Model not found"
ls -la training_summary_gpu.json 2>/dev/null || echo "Summary not found"
ls -la amsr2_gpu_sequential.log 2>/dev/null || echo "Log not found"