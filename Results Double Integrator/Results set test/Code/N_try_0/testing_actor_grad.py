import os
import os
import sys
import time
import shutil
import random
import argparse
import importlib
import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
#import tensorflow as tf
import torch
from multiprocessing import Pool
from RL import RL_AC 
from TO import TO_Casadi 
from plot_utils import PLOT
from NeuralNetwork import NN
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
import tensorflow as tf
#from tensorflow import keras
import keras.layers as layers
import keras.regularizers as regularizers
import keras.initializers as initializers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pinocchio.casadi as cpin
from types import SimpleNamespace
import math
import mpmath
import random
import numpy as np
import tensorflow as tf
import pinocchio as pin
from utils import *
import torch
import torch.nn as nn
import tensorflow as tf
from keras import layers, regularizers, initializers

class Envtf:
    def __init__(self, conf):
        '''    
        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param simu :                       (RobotSimulator instance)
            :param x_init_min :                 (float array) State lower bound initial configuration array
            :param x_init_max :                 (float array) State upper bound initial configuration array
            :param x_min :                      (float array) State lower bound vector
            :param x_max :                      (float array) State upper bound vector
            :param u_min :                      (float array) Action lower bound array
            :param u_max :                      (float array) Action upper bound array
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
    '''
        
        self.conf = conf

        self.nq = conf.nq
        self.nv = conf.nv
        self.nx = conf.nx
        self.nu = conf.na

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

    def reset(self):
        ''' Choose initial state uniformly at random '''
        state = np.zeros(self.conf.nb_state)

        time = random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1])
        for i in range(self.conf.nb_state-1): 
            state[i] = random.uniform(self.conf.x_init_min[i], self.conf.x_init_max[i]) 
        state[-1] = self.conf.dt*round(time/self.conf.dt)

        return state

    def check_ICS_feasible(self, state):
        ''' Check if ICS is not feasible '''
        # check if ee is in the obstacles
        p_ee = self.get_end_effector_position(state)

        ellipse1 = ((p_ee[0] - self.conf.XC1)**2) / ((self.conf.A1 / 2)**2) + ((p_ee[1] - self.conf.YC1)**2) / ((self.conf.B1 / 2)**2)
        ellipse2 = ((p_ee[0] - self.conf.XC2)**2) / ((self.conf.A2 / 2)**2) + ((p_ee[1] - self.conf.YC2)**2) / ((self.conf.B2 / 2)**2)
        ellipse3 = ((p_ee[0] - self.conf.XC3)**2) / ((self.conf.A3 / 2)**2) + ((p_ee[1] - self.conf.YC3)**2) / ((self.conf.B3 / 2)**2)
        
        feasible_flag = ellipse1 > 1 and ellipse2 > 1 and ellipse3 > 1

        return feasible_flag
    
    def step(self, weights, state, action):
        ''' Return next state and reward '''
        # compute next state
        state_next = self.simulate(state, action)

        # compute reward
        reward = self.reward(weights, state, action)

        return (state_next, reward)

    def simulate(self, state, action):
        ''' Simulate dynamics '''
        state_next = np.zeros(self.nx+1)

        # Simulate control action
        self.conf.simu.simulate(np.copy(state[:-1]), action, self.conf.dt, 1) ### Explicit Euler ###

        # Return next state
        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
        state_next[-1] = state[-1] + self.conf.dt
        
        return state_next
    
    def derivative(self, state, action):
        ''' Compute the derivative '''
        # Create robot model in Pinocchio with q_init as initial configuration
        q_init = state[:self.nq]
        v_init = state[self.nq:self.nx]

        # Dynamics gradient w.r.t control (1st order euler)
        pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data, np.copy(q_init), np.copy(v_init), action)       

        Fu = np.zeros((self.nx+1, self.nu))
        Fu[self.nv:-1, :] = self.conf.robot.data.Minv
        Fu[:self.nx, :] *= self.conf.dt

        if self.conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1/self.conf.state_norm_arr[:-1,None])  

        return Fu
    
    def augmented_derivative(self, state, action):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        q = state[:self.nq]
        v = state[self.nq:self.nx]
                
        # Compute Jacobians for continuous time dynamics
        Fx = np.zeros((self.conf.nb_state-1,self.conf.nb_state-1))
        Fu = np.zeros((self.conf.nb_state-1,self.conf.nb_action))

        pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data, q, v, action)

        Fx[:self.nv, :self.nv] = 0.0
        Fx[:self.nv, self.nv:self.nx] = np.identity(self.nv)
        Fx[self.nv:self.nx, :self.nv] = self.conf.robot.data.ddq_dq
        Fx[self.nv:self.nx, self.nv:self.nx] = self.conf.robot.data.ddq_dv
        Fu[self.nv:self.nx, :] = self.conf.robot.data.Minv
        
        # Convert them to discrete time
        Fx = np.identity(self.conf.nb_state-1) + self.conf.dt * Fx
        Fu *= self.conf.dt
        
        return Fx, Fu

    def simulate_batch(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        state_next = np.array([self.simulate(s, a) for s, a in zip(state, action)])

        return tf.convert_to_tensor(state_next, dtype=tf.float32)
        
    def derivative_batch(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        Fu = np.array([self.derivative(s, a) for s, a in zip(state, action)])

        return tf.convert_to_tensor(Fu, dtype=tf.float32)
    
    def get_end_effector_position(self, state, recompute=True):
        ''' Compute end-effector position '''
        q = state[:self.nq] 

        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id) 

        H = self.conf.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p
    
    def bound_control_cost(self, action):
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**10
        
        return u_cost

class DoubleIntegratortf(Envtf):
    '''
    :param cost_function_parameters :
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):

        self.conf = conf

        super().__init__(conf)

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.TARGET_STATE = self.conf.TARGET_STATE
    
    def reward(self, weights, state, action=None):
        ''' Compute reward '''
        # End-effector coordinates
        x_ee, y_ee = [self.get_end_effector_position(state)[i] for i in range(2)]

        # Penalties for the ellipses representing the obstacle
        ell1_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_rew = np.math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        if action is not None:
            u_cost = self.bound_control_cost(action)
        else:
            u_cost = 0

        dist_cost = (x_ee-self.TARGET_STATE[0])**2 + (y_ee-self.TARGET_STATE[1])**2

        r = self.scale*(- weights[0]*dist_cost + weights[1]*peak_rew - weights[3]*ell1_cost - weights[4]*ell2_cost - weights[5]*ell3_cost - weights[6]*u_cost + self.offset)
        
        return r
    
    def reward_batch(self, weights, state, action):
        ''' Compute reward using tensors. Batch-wise computation '''
        partial_reward = np.array([self.reward(w, s) for w, s in zip(weights, state)])

        # Redefine action-related cost in tensorflow version
        u_cost = tf.reduce_sum((action**2 + self.conf.w_b*(action/self.conf.u_max)**10),axis=1) 

        r = self.scale*(- weights[:,6]*u_cost) + tf.convert_to_tensor(partial_reward, dtype=tf.float32)

        return tf.reshape(r, [r.shape[0], 1])

# Import configuration file and environment file
system_map = {
    'single_integrator': ('conf_single_integrator', 'SingleIntegrator', 'SingleIntegrator_CAMS'),
    'double_integrator': ('conf_double_integrator', 'DoubleIntegrator', 'DoubleIntegrator_CAMS'),
    'car':               ('conf_car', 'Car', 'Car_CAMS'),
    'car_park':          ('conf_car_park', 'CarPark', 'CarPark_CAMS'),
    'manipulator':       ('conf_manipulator', 'Manipulator', 'Manipulator_CAMS'),
    'ur5':               ('conf_ur5', 'UR5', 'UR5_CAMS')
}
try:
    conf_module, env_class, env_TO_class = system_map['double_integrator']
    conf = importlib.import_module(conf_module)
    Environment = getattr(importlib.import_module('environment'), env_class)
    Environment_TO = getattr(importlib.import_module('environment_TO'), env_TO_class)
except KeyError:
    print('System {} not found'.format('N/A'))
    sys.exit()

#from robot_utils import RobotSimulator, RobotWrapper
#from types import SimpleNamespace

class LinearLayerL1L2(nn.Module):
    '''
    Linear Layer with L1 and L2 regularization applied afterwards. 4 separate weights are used 
    for each of L1/L2 kernel/bias regularization. This is done to replace the calls to 
    tf.keras.regularizers which doesn't have a PyTorch equivalent
    '''
    def __init__(self, in_features, out_features, kreg_l1, kreg_l2, breg_l1, breg_l2):
        super(LinearLayerL1L2, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.kreg_l1 = kreg_l1
        self.kreg_l2 = kreg_l2
        self.breg_l1 = breg_l1
        self.breg_l2 = breg_l2

    def forward(self, x):
        x = self.linear(x)

        # Regularization for weights
        if self.kreg_l1 > 0:
            l1_regularization_w = self.kreg_l1 * torch.sum(torch.abs(self.linear.weight))
            x += l1_regularization_w
        if self.kreg_l2 > 0:
            l2_regularization_w = self.kreg_l2 * torch.sum(torch.pow(self.linear.weight, 2))
            x += l2_regularization_w

        # Regularization for biases
        if self.breg_l1 > 0:
            l1_regularization_b = self.breg_l1 * torch.sum(torch.abs(self.linear.bias))
            x += l1_regularization_b
        if self.breg_l2 > 0:
            l2_regularization_b = self.breg_l2 * torch.sum(torch.pow(self.linear.bias, 2))
            x += l2_regularization_b

        return x


class SineActivation(nn.Module):
    '''
    Sinusoidal activation function with weight w0
    '''
    def __init__(self, w0=1.0):
        super(SineActivation, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class SineRepLayer(nn.Module):
    '''
    This is the PyTorch Equivalent of the SinusodialRepresentationDense class
    from siren.py in the tf_siren package. Yes, sinusoidal is misspelled in that package
    and hence in this class as well. This class represents a linear layer initialized according to the 
    paper 'Implicit Neural Representations with Periodic Activation Functions', followed by a sinusoidal
    activation function
    '''
    #weights initialized with uniform
    #biases initialized with he_uniform (see tf/keras fr details)
    
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
def create_actor_torch():
    ''' Create actor NN '''
    model = nn.Sequential(
        #LinearLayerL1L2(conf.nb_state, 64, 1e-2,1e-2, 1e-2,1e-2),
        nn.Linear(conf.nb_state, 64),
        nn.LeakyReLU(),
        #LinearLayerL1L2(64, 64, 1e-2,1e-2, 1e-2,1e-2),
        nn.Linear(64,64),
        nn.LeakyReLU(),
        #LinearLayerL1L2(64, conf.nb_action, 1e-2,1e-2, 1e-2,1e-2)
        nn.Linear(64, conf.nb_action)
    )
    for layer in model:
            try:
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 1)
            except:
                continue
    return model
def create_critic_elu_torch(): 
        ''' Create critic NN - elu'''
        model = nn.Sequential(
            nn.Linear(conf.nb_state, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        for layer in model:
            try:
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 1)
            except:
                continue
        return model
def create_actor_tf():
    ''' Create actor NN '''
    inputs = layers.Input(shape=(conf.nb_state,))
    
    lay1 = layers.Dense(
        64,
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Ones(),
        kernel_regularizer=regularizers.l1_l2(1e-2, 1e-2),
        bias_regularizer=regularizers.l1_l2(1e-2, 1e-2)
    )(inputs)
    leakyrelu1 = layers.LeakyReLU()(lay1)
    
    lay2 = layers.Dense(
        64,
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Ones(),
        kernel_regularizer=regularizers.l1_l2(1e-2, 1e-2),
        bias_regularizer=regularizers.l1_l2(1e-2, 1e-2)
    )(leakyrelu1)
    leakyrelu2 = layers.LeakyReLU()(lay2)
    
    outputs = layers.Dense(
        conf.nb_action,
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Ones(),
        kernel_regularizer=regularizers.l1_l2(1e-2, 1e-2),
        bias_regularizer=regularizers.l1_l2(1e-2, 1e-2)
    )(leakyrelu2)

    model = tf.keras.Model(inputs, outputs)
    return model
def create_critic_elu_tf(): 
    ''' Create critic NN - elu '''
    state_input = layers.Input(shape=(conf.nb_state,))

    state_out1 = layers.Dense(
        16,
        activation='elu',
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Ones()
    )(state_input)
    
    state_out2 = layers.Dense(
        32,
        activation='elu',
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Ones()
    )(state_out1)
    
    out_lay1 = layers.Dense(
        256,
        activation='elu',
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Ones()
    )(state_out2)
    
    out_lay2 = layers.Dense(
        256,
        activation='elu',
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Ones()
    )(out_lay1)
    
    outputs = layers.Dense(
        1,
        kernel_initializer=initializers.Ones(),
        bias_initializer=initializers.Ones()
    )(out_lay2)

    model = tf.keras.Model([state_input], outputs)
    return model 
def eval_torch(NN, input):
    ''' Compute the output of a NN given an input '''
    if not torch.is_tensor(input):
        if isinstance(input, list):
            input = np.array(input)
        input = torch.tensor(input, dtype=torch.float32)

    if conf.NORMALIZE_INPUTS:
        input = normalize_tensor_torch(input, torch.tensor(conf.state_norm_arr))

    return NN(input)
def eval_tf(NN, input):
    ''' Compute the output of a NN given an input '''
    if not tf.is_tensor(input):
        input = tf.convert_to_tensor(input, dtype=tf.float32)

    if conf.NORMALIZE_INPUTS:
        input = normalize_tensor_tf(input, conf.state_norm_arr)

    return NN(input, training=True)
def custom_logarithm_torch(input):
    # Calculate the logarithms based on the non-zero condition
    positive_log = torch.log(torch.maximum(input, torch.tensor(1e-7)) + 1)
    negative_log = -torch.log(torch.maximum(-input, torch.tensor(1e-7)) + 1)

    # Use the appropriate logarithm based on the condition
    result = torch.where(input > 0, positive_log, negative_log)

    return result    
def custom_logarithm_tf(input):
    # Calculate the logarithms based on the non-zero condition
    positive_log = tf.math.log(tf.math.maximum(input, 1e-7) + 1)
    negative_log = -tf.math.log(tf.math.maximum(-input, 1e-7) + 1)

    # Use the appropriate logarithm based on the condition
    result = tf.where(input > 0, positive_log, negative_log)

    return result    
def normalize_tensor_tf(state, state_norm_arr):
    ''' Retrieve state from normalized state - tensor '''
    state_norm_time = tf.concat([tf.zeros([state.shape[0], state.shape[1]-1]), tf.reshape(((state[:,-1]) / state_norm_arr[-1])*2 - 1,[state.shape[0],1])],1)
    state_norm_no_time = state / state_norm_arr
    mask = tf.concat([tf.ones([state.shape[0], state.shape[1]-1]), tf.zeros([state.shape[0], 1])],1)
    state_norm = state_norm_no_time * mask + state_norm_time * (1 - mask)

    return state_norm
def normalize_tensor_torch(state, state_norm_arr):
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
    return state_norm
class WeightedMSELoss(torch.nn.Module):
    '''
    Weighted MSE Loss class to match tensorflow functionality with no reduction.
    '''
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
dummy = True #this replaces self.conf.MC
w_S = 1e-2

def compute_actor_grad_tf(actor_model, critic_model, state_batch, term_batch, batch_size):
    ''' Compute the gradient of the actor NN '''
    if batch_size == None:
        batch_size = conf.BATCH_SIZE

    actions = eval_tf(actor_model, state_batch)

    # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
    state_next_tf, ds_next_da = envtf.simulate_batch(state_batch.numpy(), actions.numpy()) , envtf.derivative_batch(state_batch.numpy(), actions.numpy())
    #state_next_tf = tf.convert_to_tensor(state_next_tf.detach().numpy())
    with tf.GradientTape() as tape:
        tape.watch(state_next_tf)
        critic_value_next = eval_tf(critic_model,state_next_tf) 

    # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)                                           
    dV_ds_next = tape.gradient(critic_value_next, state_next_tf)

    cost_weights_terminal_reshaped = np.reshape(conf.cost_weights_terminal,[1,len(conf.cost_weights_terminal)])
    cost_weights_running_reshaped = np.reshape(conf.cost_weights_running,[1,len(conf.cost_weights_running)])
    with tf.GradientTape() as tape1:
        tape1.watch(actions)
        rewards_tf = envtf.reward_batch(term_batch.dot(cost_weights_terminal_reshaped) + (1-term_batch).dot(cost_weights_running_reshaped), state_batch.numpy(), actions)

    # dr_da = gradient of reward r(s,a) w.r.t. policy's action a
    dr_da = tape1.gradient(rewards_tf, actions, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    dr_da_reshaped = tf.reshape(dr_da, (batch_size, 1, conf.nb_action))
    
    # dr_ds' + dV_ds' (note: dr_ds' = 0)
    dQ_ds_next = tf.reshape(dV_ds_next, (batch_size, 1, conf.nb_state))        
    
    # (dr_ds' + dV_ds')*ds'_da
    dQ_ds_next_da = tf.matmul(dQ_ds_next, ds_next_da)
    
    # (dr_ds' + dV_ds')*ds'_da + dr_da
    dQ_da = dQ_ds_next_da + dr_da_reshaped

    # Now let's multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a 
    # and then let's autodifferentiate w.r.t theta_A (actor NN's parameters) to finally get -dQ/dtheta_A 
    with tf.GradientTape() as tape:
        tape.watch(actor_model.trainable_variables)
        actions = eval_tf(actor_model, state_batch)
        
        actions_reshaped = tf.reshape(actions,(batch_size, conf.nb_action,1))
        dQ_da_reshaped = tf.reshape(dQ_da,(batch_size,1, conf.nb_action))    
        Q_neg = tf.matmul(-dQ_da_reshaped,actions_reshaped) 
        
        # Also here we need a scalar so we compute the mean -Q across the batch
        mean_Qneg = tf.math.reduce_mean(Q_neg)

    # Gradients of the actor loss w.r.t. actor's parameters
    print(mean_Qneg)
    actor_grad = tape.gradient(mean_Qneg, actor_model.trainable_variables)

    return actor_grad

def compute_actor_grad_torch(actor_model, critic_model, state_batch, term_batch, batch_size=None):
    ''' Compute the gradient of the actor NN '''
    #critic_model.eval()
    #actor_model.train()
    if batch_size is None:
        batch_size = conf.BATCH_SIZE

    actions = eval_torch(actor_model, state_batch)

    # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
    act_np = actions.detach().numpy()
    state_next_tf, ds_next_da = env.simulate_batch(state_batch.detach().numpy(), act_np), env.derivative_batch(state_batch.detach().numpy(), act_np)
    
    #state_next_tf = torch.tensor(state_next_tf, requires_grad=True, dtype=torch.float32)
    state_next_tf = state_next_tf.clone().detach().to(dtype=torch.float32).requires_grad_(True)
    #ds_next_da = torch.tensor(ds_next_da, requires_grad=True, dtype=torch.float32)
    ds_next_da = ds_next_da.clone().detach().to(dtype=torch.float32).requires_grad_(True)

    # Compute critic value at the next state
    critic_value_next = eval_torch(critic_model, state_next_tf)

    # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)
    
    dV_ds_next = torch.autograd.grad(outputs=critic_value_next, inputs=state_next_tf,
                                     grad_outputs=torch.ones_like(critic_value_next),
                                     create_graph=True)[0]

    cost_weights_terminal_reshaped = torch.tensor(conf.cost_weights_terminal, dtype=torch.float32).reshape(1, -1)
    cost_weights_running_reshaped = torch.tensor(conf.cost_weights_running, dtype=torch.float32).reshape(1, -1)

    # Compute rewards
    #term_batch = torch.tensor(term_batch, dtype=torch.float32)
    #reward_inputs = term_batch @ cost_weights_terminal_reshaped + (1 - term_batch) @ cost_weights_running_reshaped
    #reward_inputs = reward_inputs.numpy()
    #rewards_tf = env.reward_batch(reward_inputs, state_batch.detach().numpy(), actions.detach().numpy())
    #rewards_tf = env.reward_batch(reward_inputs.detach().numpy(), state_batch.detach().numpy(), actions)
    #actions = torch.tensor(actions, requires_grad=True, dtype=torch.float32)
    rewards_tf = env.reward_batch(term_batch.dot(cost_weights_terminal_reshaped) + (1-term_batch).dot(cost_weights_running_reshaped), state_batch.detach().numpy(), actions)

    # dr_da = gradient of reward r(s,a) w.r.t. policy's action a
    #rewards_tf = torch.tensor(rewards_tf, requires_grad=True)
    dr_da = torch.autograd.grad(outputs=rewards_tf, inputs=actions,
                                grad_outputs=torch.ones_like(rewards_tf),
                                create_graph=True)[0]

    dr_da_reshaped = dr_da.view(batch_size, 1, conf.nb_action)

    # dr_ds' + dV_ds' (note: dr_ds' = 0)
    dQ_ds_next = dV_ds_next.view(batch_size, 1, conf.nb_state)

    # (dr_ds' + dV_ds')*ds'_da
    dQ_ds_next_da = torch.bmm(dQ_ds_next, ds_next_da)

    # (dr_ds' + dV_ds')*ds'_da + dr_da
    dQ_da = dQ_ds_next_da + dr_da_reshaped

    # Multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a
    actions = eval_torch(actor_model, state_batch)
    actions_reshaped = actions.view(batch_size, conf.nb_action, 1)
    dQ_da_reshaped = dQ_da.view(batch_size, 1, conf.nb_action)
    #Q_neg = torch.bmm(-dQ_da_reshaped, actions_reshaped)
    Q_neg = torch.matmul(-dQ_da_reshaped, actions_reshaped)

    # Compute the mean -Q across the batch
    mean_Qneg = Q_neg.mean()

    # Gradients of the actor loss w.r.t. actor's parameters
    #actor_grad = torch.autograd.grad(outputs=mean_Qneg, inputs=actor_model.parameters(),
    #                        grad_outputs=torch.ones_like(mean_Qneg),
    #                        create_graph=True)[0]
    actor_model.zero_grad()
    #print(mean_Qneg)
    #actor_grad = torch.autograd.grad(mean_Qneg, actor_model.parameters())
    #actor_model.zero_grad()
    mean_Qneg.backward()
    actor_grad = [param.grad for param in actor_model.parameters()]

    return actor_grad



#self, actor_model, critic_model, state_batch, term_batch, batch_size
np.random.seed(200)
torch.manual_seed(100)
tf.random.set_seed(100)
# Initialize models
actor_model_tf = create_actor_tf()
critic_model_tf = create_critic_elu_tf()

actor_model_torch = create_actor_torch()
critic_model_torch = create_critic_elu_torch()

# Generate test data
state_batch_np = np.random.rand(conf.BATCH_SIZE, conf.nb_state).astype(np.float32)
state_batch_torch = torch.tensor(state_batch_np, requires_grad=True)
state_batch_tf = tf.convert_to_tensor(state_batch_np)

term_batch_np = np.random.rand(conf.BATCH_SIZE, 1).astype(np.float32)
#term_batch_torch = torch.tensor(term_batch_np, requires_grad=True)
#term_batch_tf = tf.convert_to_tensor(term_batch_np)

#print(actor_model_torch(state_batch_torch))
#print(actor_model_tf(state_batch_tf))

### Create instances of the used classes ###
env = Environment(conf)
#Environmenttf = getattr(importlib.import_module('environmenttf'), 'DoubleIntegratortf')
envtf = DoubleIntegratortf(conf)

actor_grad_tf = compute_actor_grad_tf(actor_model_tf, critic_model_tf, state_batch_tf, term_batch_np, batch_size=None)
print(actor_grad_tf)
print('-'*50)
actor_grad_torch = compute_actor_grad_torch(actor_model_torch, critic_model_torch, state_batch_torch, term_batch_np, batch_size=None)
print(actor_grad_torch)


# Compare results
def compare_tensors(tensor_tf, tensor_torch):
    tensor_torch_np = tensor_torch.detach().numpy() if isinstance(tensor_torch, torch.Tensor) else np.array(tensor_torch)
    return np.allclose(tensor_tf.numpy(), tensor_torch_np, atol=1e-5)

# Compare gradients
#print(actor_grad_tf)
print('-'*40)
#print(actor_grad_torch)
for grad_tf, grad_torch in zip(actor_grad_tf, actor_grad_torch):
    print("Gradient Comparison:", compare_tensors(grad_tf, torch.t(grad_torch)))
'''
np.random.seed(100)
for i in range(1000):
    # Generate random dimensions
    num_dimensions = np.random.randint(1, 6)  # Random number of dimensions between 1 and 5
    dimensions = [np.random.randint(1, 10) for _ in range(num_dimensions)]  # Random size for each dimension

    # Create a random NumPy array with these dimensions
    random_array = np.random.rand(*dimensions)
    res_torch = custom_logarithm_torch(torch.tensor(random_array))
    res_tf = custom_logarithm_tf(tf.constant(random_array))
    if not np.allclose(res_tf.numpy(), res_torch.detach().numpy(), atol=1e-5):
        print(f'Test {i} failed')
    else:
        print(f'Test {i} passed')
    '''
