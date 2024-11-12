import torch
from torch.utils import cpp_extension

# Check if CUDA is available
print(f'Cuda is available: {torch.cuda.is_available()}')


# Load the shared library containing the CUDA kernel
kernel_lib = cpp_extension.load(name='kernel', sources=['kernel.cu'], verbose=True)

# Create a tensor on the GPU
tensor = torch.zeros((3,3), dtype=torch.float32, device='cuda:0')

# Print the original value
print("Original value:", tensor)

# Print tensor address before call
print(f"Memory address before kernel call: {tensor.data_ptr()}")


# Call the CUDA kernel multiple times in a loop
for _ in range(3):
    kernel_lib.add_one_at_address(tensor)
    tensor[0,0] += 1

# Check the result after modifying the tensor's value via the kernel
print("Modified value:", tensor)

# Print tensor address after call:
print(f"Memory address after kernel call: {tensor.data_ptr()}")

