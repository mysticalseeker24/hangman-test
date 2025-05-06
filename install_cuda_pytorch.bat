@echo off
echo Installing CUDA-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo Installation complete. Please restart your Jupyter notebook kernel.
