import torch
import torch.nn as nn
import tensorflow as tf
import importlib
from TO import TO_Casadi
import numpy as np
import keras
from keras import layers, initializers, regularizers
from tf_siren import SinusodialRepresentationDense

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


class SineActivation(nn.Module):
    '''
    Sinusoidal activation function with weight w0
    '''
    #Tested Successfully#
    def __init__(self, w0=1.0):
        super(SineActivation, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class SineRepLayer(nn.Module):
    '''
    This is the PyTorch Equivalent of the SinusodialRepresentationDense class
    from siren.py in the tf_siren package.This class represents a linear layer initialized according to the 
    paper 'Implicit Neural Representations with Periodic Activation Functions', followed by a sinusoidal
    activation function
    '''
    #Tested Successfully#
    
    def __init__(self, in_features, out_features, w0=1.0, c=6.0, use_bias=True):
        model = nn.Sequential(
            nn.Linear(in_features, out_features),
            SineActivation(w0)
        )
        super(SineRepLayer, self).__init__()
        self.w0 = w0
        self.c = c
        self.scale = c / (3.0 * w0 * w0)

        self.model = model

        # weights initialization
        fan_in, _ = _compute_fans(self.model[0].weight.shape)
        self.scale /= max(1.0, fan_in)
        limit = np.sqrt(3.0 * self.scale)
        #NOTE: MAKE THIS USE A GENERATOR FOR SEED USAGE
        nn.init.uniform_(self.model[0].weight, -limit, limit)
        
        if use_bias:
            #biases initialization
            nn.init.uniform_(self.model[0].bias, -np.sqrt(6/fan_in), np.sqrt(6/fan_in))

    def forward(self, x):
        return self.model(x)

def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
      shape: Integer shape tuple or TF tensor shape.

    Returns:
      A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)

def create_critic_sine_torch():
    #Tested Successfully# 
    ''' Create critic NN - elu'''
    model = nn.Sequential(
        SineRepLayer(conf.nb_state, 64),
        SineRepLayer(64, 64),
        SineRepLayer(64, 128),
        SineRepLayer(128, 128),
        nn.Linear(128, 1)
    )
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    return model
def create_actor_torch():
    ''' Create actor NN '''
    #Tested Successfully#
    model = nn.Sequential(
        nn.Linear(conf.nb_state, conf.NH1),
        nn.LeakyReLU(negative_slope=0.3),
        nn.Linear(conf.NH1, conf.NH2),
        nn.LeakyReLU(negative_slope=0.3),
        nn.Linear(conf.NH2, conf.nb_action)
    )
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    return model

def create_actor_tf():
    ''' Create actor NN '''
    inputs = layers.Input(shape=(conf.nb_state,))
    
    lay1 = layers.Dense(conf.NH1,kernel_regularizer=regularizers.l1_l2(conf.kreg_l1_A,conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(conf.breg_l1_A,conf.breg_l2_A))(inputs)                                        
    leakyrelu1 = layers.LeakyReLU()(lay1)
    lay2 = layers.Dense(conf.NH2, kernel_regularizer=regularizers.l1_l2(conf.kreg_l1_A,conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(conf.breg_l1_A,conf.breg_l2_A))(leakyrelu1)                                           
    leakyrelu2 = layers.LeakyReLU()(lay2)
    outputs = layers.Dense(conf.nb_action, kernel_regularizer=regularizers.l1_l2(conf.kreg_l1_A,conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(conf.breg_l1_A,conf.breg_l2_A))(leakyrelu2) 

    model = tf.keras.Model(inputs, outputs)

    return model
def create_critic_sine_tf(): 
    ''' Create critic NN - elu'''
    state_input = layers.Input(shape=(conf.nb_state,))
    
    state_out1 = SinusodialRepresentationDense(64, activation='sine')(state_input) 
    state_out2 = SinusodialRepresentationDense(64, activation='sine')(state_out1) 
    out_lay1 = SinusodialRepresentationDense(128, activation='sine')(state_out2)
    out_lay2 = SinusodialRepresentationDense(128, activation='sine')(out_lay1)
    
    outputs = layers.Dense(1)(out_lay2)

    model = tf.keras.Model([state_input], outputs)

    return model   



torch_actor_path = '/home/a2rlab/cacto-pytorch/Results Double Integrator/Results set test/NNs/N_try_69/actor_final.pth'
torch_target_critic_path = '/home/a2rlab/cacto-pytorch/Results Double Integrator/Results set test/NNs/N_try_69/target_critic_final.pth'
torch_critic_path = '/home/a2rlab/cacto-pytorch/Results Double Integrator/Results set test/NNs/N_try_69/critic_final.pth'
torch_critic_model = create_critic_sine_torch()
torch_target_critic = create_critic_sine_torch()
torch_actor_model = create_actor_torch()


tf_actor_path = '/home/a2rlab/cacto/Results Double Integrator/Results set test/NNs/N_try_53/actor_final.h5'
tf_critic_path = '/home/a2rlab/cacto/Results Double Integrator/Results set test/NNs/N_try_53/critic_final.h5'
tf_target_critic_path = '/home/a2rlab/cacto/Results Double Integrator/Results set test/NNs/N_try_53/target_critic_final.h5'
tf_critic_model = create_critic_sine_tf()
tf_target_critic = create_critic_sine_tf()
tf_actor_model = create_actor_tf()


x = [a for a in torch_actor_model.parameters()]
#print(x)
#print('-'*40)
#print(tf_actor_model.get_weights())
#print('-'*40)
summ = 0
for a in x:
    summ += torch.sum(a)
print(summ)
summ2 = 0
for a in tf_actor_model.get_weights():
    summ2 += tf.reduce_sum(a)
print(summ2)

# Load saved weights
torch_critic_model.load_state_dict(torch.load(torch_critic_path))
torch_target_critic.load_state_dict(torch.load(torch_target_critic_path))
torch_actor_model.load_state_dict(torch.load(torch_actor_path))
tf_critic_model.load_weights(tf_critic_path)
tf_target_critic.load_weights(tf_target_critic_path)
tf_actor_model.load_weights(tf_actor_path)

x = [a for a in torch_actor_model.parameters()]
#print(x)
print('-'*40)
#print(tf_actor_model.get_weights())
#print('-'*40)
summ = 0
for a in x:
    summ += torch.sum(a)
print(summ)
summ2 = 0
for a in tf_actor_model.get_weights():
    summ2 += tf.reduce_sum(a)
print(summ2)
