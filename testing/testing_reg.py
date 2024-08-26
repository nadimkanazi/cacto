import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import importlib
from TO import TO_Casadi 
import tensorflow as tf
import torch.nn as nn
import pinocchio.casadi as cpin
import pinocchio as pin
import torch.nn as nn
from keras import layers, regularizers
import torch
import numpy as np
from tf_siren import SinusodialRepresentationDense
from torchsummary import summary

# def _compute_fans(shape):
#     """Computes the number of input and output units for a weight shape.

#     Args:
#       shape: Integer shape tuple or TF tensor shape.

#     Returns:
#       A tuple of integer scalars (fan_in, fan_out).
#     """
#     if len(shape) < 1:  # Just to avoid errors for constants.
#         fan_in = fan_out = 1
#     elif len(shape) == 1:
#         fan_in = fan_out = shape[0]
#     elif len(shape) == 2:
#         fan_in = shape[0]
#         fan_out = shape[1]
#     else:
#         # Assuming convolution kernels (2D, 3D, or more).
#         # kernel shape: (..., input_depth, depth)
#         receptive_field_size = 1
#         for dim in shape[:-2]:
#             receptive_field_size *= dim
#         fan_in = shape[-2] * receptive_field_size
#         fan_out = shape[-1] * receptive_field_size
#     return int(fan_in), int(fan_out)

# class SineActivation(nn.Module):
#     '''
#     Sinusoidal activation function with weight w0
#     '''
#     def __init__(self, w0=1.0):
#         super(SineActivation, self).__init__()
#         self.w0 = w0

#     def forward(self, x):
#         return torch.sin(self.w0 * x)

# class SineRepLayer(nn.Module):
#     '''
#     This is the PyTorch Equivalent of the SinusodialRepresentationDense class
#     from siren.py in the tf_siren package. Yes, sinusoidal is misspelled in that package
#     and hence in this class as well. This class represents a linear layer initialized according to the 
#     paper 'Implicit Neural Representations with Periodic Activation Functions', followed by a sinusoidal
#     activation function
#     '''
#     #weights initialized with uniform
#     #biases initialized with he_uniform (see tf/keras fr details)
    
#     def __init__(self, in_features, out_features, w0=1.0, c=6.0, use_bias=True):
#         model = nn.Sequential(
#             nn.Linear(in_features, out_features),
#             SineActivation(w0)
#         )
#         super(SineRepLayer, self).__init__()
#         self.w0 = w0
#         self.c = c
#         self.scale = c / (3.0 * w0 * w0)

#         self.model = model

#         # weights initialization
#         fan_in, _ = _compute_fans(self.model[0].weight.shape)
#         self.scale /= max(1.0, fan_in)
#         limit = np.sqrt(3.0 * self.scale)
#         #NOTE: MAKE THIS USE A GENERATOR FOR SEED USAGE
#         #nn.init.uniform_(self.model[0].weight, -limit, limit)
#         nn.init.constant_(self.model[0].weight, 1)
        
#         if use_bias:
#             #biases initialization
#             #nn.init.uniform_(self.model[0].bias, -np.sqrt(6/fan_in), np.sqrt(6/fan_in))
#             nn.init.constant_(self.model[0].bias, 1)

#     def forward(self, x):
#         return self.model(x)

# a = SinusodialRepresentationDense(64, activation='sine',kernel_initializer=initializers.Ones(), bias_initializer=initializers.Ones())
# b = SineRepLayer(64, 64)

# vec = np.random.rand(1,64)
# vec_tf = tf.convert_to_tensor(vec)
# vec_torch = torch.tensor(vec, dtype=torch.float)
# print(a(vec_tf))
# print(b(vec_torch))

system_map = {
    'single_integrator': ('conf_single_integrator', 'SingleIntegrator', 'SingleIntegrator_CAMS'),
    'double_integrator': ('conf_double_integrator', 'DoubleIntegrator', 'DoubleIntegrator_CAMS'),
    'car':               ('conf_car', 'Car', 'Car_CAMS'),
    'car_park':          ('conf_car_park', 'CarPark', 'CarPark_CAMS'),
    'manipulator':       ('conf_manipulator', 'Manipulator', 'Manipulator_CAMS'),
    'ur5':               ('conf_ur5', 'UR5', 'UR5_CAMS')
}

conf_module, env_class, env_TO_class = system_map['double_integrator']
conf = importlib.import_module(conf_module)
#Environment = getattr(importlib.import_module('environment'), env_class)
#Environment_TO = getattr(importlib.import_module('environment_TO'), env_TO_class)

class WeightedMSELoss(torch.nn.Module):
    '''
    Weighted MSE Loss class to match tensorflow functionality with no reduction.
    '''
    #Tested Successfully#
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, inputs, targets, weights=None):
        # Ensure inputs and targets are the same shape
        if inputs.shape != targets.shape:
            raise ValueError("Inputs and targets must have the same shape")
        
        # Compute MSE loss
        #mse_loss = torch.pow((inputs - targets)*weights, 2)
        mse_loss = torch.pow(inputs - targets, 2)
        
        if weights is not None:
            if weights.shape != inputs.shape:
                weights.expand(inputs.shape)
                #raise ValueError("Weights must have the same shape as inputs and targets")
            mse_loss = mse_loss * weights
        
        return torch.mean(mse_loss)  # Return the mean of the loss

# Example inputs
inputs_tf = tf.constant([1.0, 2.0, 3.0], dtype=tf.float16)
targets_tf = tf.constant([1.5, 2.5, 3.5], dtype=tf.float16)
weights_tf = tf.constant([0.1, 0.2, 0.3], dtype=tf.float16)
weights_tf = tf.reshape(weights_tf, (1,-1))

# Calculate loss
mse_loss_tf = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
loss_tf = mse_loss_tf(targets_tf, inputs_tf, sample_weight=weights_tf)
loss_tf = tf.reduce_mean(loss_tf)
print(f"TensorFlow Weighted MSE Loss: {loss_tf.numpy()}")

inputs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
targets = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float16)
weights = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16)

# Instantiate loss function
mse_loss_fn = WeightedMSELoss()

# Calculate loss
loss = mse_loss_fn(inputs, targets, weights)
print(f"PyTorch Weighted MSE Loss: {loss.item()}")