import torch
from torch.utils import cpp_extension

# Check if CUDA is available
print(torch.cuda.is_available())

# Load the shared library that contains the CUDA kernel
kernel_lib = cpp_extension.load(name='kernel', sources=['kernel.cu'], verbose=True)

# Create a tensor on the GPU
tensor = torch.zeros((), dtype=torch.float32, device='cuda:0')

# Print the original value
print("Original value:", tensor.item())

# Call the CUDA kernel
# This will directly pass the tensor to the kernel function
kernel_lib.add_one_at_address(tensor)

# Check the result after modifying the tensor's value via the kernel
print("Modified value:", tensor.item())


