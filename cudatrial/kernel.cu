#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to add 1 to the tensor value at a given address
__global__ void add_one_at_address(float *ptr) {
    *ptr += 1.0f;
}

// Wrapper function for the kernel
void add_one_at_address_wrapper(torch::Tensor tensor) {
    // Ensure that the tensor is on the CUDA device
    AT_ASSERTM(tensor.is_cuda(), "Tensor must be on CUDA device");

    // Getting a pointer to the memory of the tensor
    float *ptr = tensor.data_ptr<float>();

    // Launch the CUDA kernel with 1 block and 1 thread
    add_one_at_address<<<1, 1>>>(ptr);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();
}

// Python module initialization
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one_at_address", &add_one_at_address_wrapper, "Add one to a tensor at a memory address");
}


