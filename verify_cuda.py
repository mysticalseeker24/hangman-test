import torch
import os

def check_cuda():
    print("===== CUDA INFORMATION =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Test a simple tensor operation on GPU
        print("\nRunning simple GPU test...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        z = x @ y  # matrix multiplication
        end.record()
        torch.cuda.synchronize()
        print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")
        print("GPU test completed successfully!")
        
        # Memory info
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Check your NVIDIA drivers and PyTorch installation.")
        print("Run the install_cuda_pytorch.bat script to install the CUDA version of PyTorch.")

    print("\n===== NOTEBOOK MODIFICATION GUIDE =====")
    print("1. Make sure to restart your Jupyter kernel after installing CUDA PyTorch")
    print("2. Verify your model tensors are moved to cuda device: tensor.to(device) or tensor.cuda()")
    print("3. Ensure all inputs to model are on the same device: input.to(device)")
    print("4. Check that the model itself is on GPU: model.to(device)")
    print("5. For batches, prefer using larger batch sizes on GPU to maximize throughput")

if __name__ == "__main__":
    check_cuda()
