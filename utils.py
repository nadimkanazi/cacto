import torch
import numpy as np

'''
Note: the only 2 functions from this file that are called in main.py are array2tensor and normalize_tensor.
Therefore these were the only 2 functions that were changed and tested.
'''

def array2tensor(array):
    if isinstance(array, list):
        array = np.array(array)
    elif torch.is_tensor(array):
        return array
    return torch.unsqueeze(torch.tensor(array, dtype=torch.float16), 0)

def de_normalize_tensor(state, state_norm_arr):
    ''' Retrieve state from normalized state - tensor '''
    state_time = torch.cat([torch.zeros([state.shape[0], state.shape[1] - 1]), torch.reshape((state[:, -1] + 1) * state_norm_arr[-1] / 2, (state.shape[0], 1))], dim=1)
    state_no_time = state * state_norm_arr
    mask = torch.cat([torch.ones([state.shape[0], state.shape[1] - 1]), torch.zeros([state.shape[0], 1])], dim=1)
    state_not_norm = state_no_time * mask + state_time * (1 - mask)
    
    return state_not_norm

def normalize_tensor(state, state_norm_arr):
    ''' Retrieve state from normalized state - tensor '''
    state_norm_time = torch.cat([
        torch.zeros([state.shape[0], state.shape[1] - 1]),
        torch.reshape((state[:, -1] / state_norm_arr[-1]) * 2 - 1, (state.shape[0], 1))
    ], dim=1)
    
    state_norm_no_time = state / state_norm_arr
    mask = torch.cat([
        torch.ones([state.shape[0], state.shape[1] - 1]),
        torch.zeros([state.shape[0], 1])
    ], dim=1)
    
    state_norm = state_norm_no_time * mask + state_norm_time * (1 - mask)
    return state_norm.to(torch.float16)

def de_normalize(state, state_norm_arr):
    ''' Retrieve state from normalized state '''
    state_not_norm  = np.empty_like(state)
    state_not_norm[:-1] = state[:-1] * state_norm_arr[:-1]
    state_not_norm[-1] = (state[-1] + 1) * state_norm_arr[-1]/2

    return state_not_norm

def normalize(state, state_norm_arr):
    ''' Normalize state '''
    state_norm  = np.empty_like(state)
    state_norm = state / state_norm_arr
    state_norm[-1] = state_norm[-1] * 2 -1

    return state_norm

'''
arr = np.random.rand(64,6)
print(arr.shape)
a = torch.tensor(arr)
b = tf.Variable(arr, dtype=tf.float16)
n = np.array([10, 3, 3.14, 10, 3.14/6, 5])
#print(a)
#print(b)
a = normalize_tensor_torch(a, n)
b = normalize_tensor_tf(b, n)
a_np = a.detach().numpy()
b_np = b.numpy()
# Compare the tensors
mask = np.isclose(a_np, b_np, atol=1e-6)
# Print comparison results
print("Are the tensors equal? ", mask.all())
# Print differing elements if there are any
if not mask.all():
    print("Differences found:")
    print("PyTorch tensor elements where they differ:", a_np[~mask])
    print("TensorFlow tensor elements where they differ:", b_np[~mask])
else:
    print("No differences found.")
'''
