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
import uuid
import math
import numpy as np
#import tensorflow as tf
import torch
from NeuralNetwork import NN
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tf_siren import SinusodialRepresentationDense

from utils import normalize_tensor

class NN_tf:

    def __init__(self, env, conf, w_S=0):
        '''    
        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)

            :param NH1:                         (int) 1st hidden layer size
            :param NH2:                         (int) 2nd hidden layer size
            :param kreg_l1_A :                  (float) Weight of L1 regularization in actor's network - kernel  
            :param kreg_l2_A :                  (float) Weight of L2 regularization in actor's network - kernel  
            :param breg_l1_A :                  (float) Weight of L2 regularization in actor's network - bias  
            :param breg_l2_A :                  (float) Weight of L2 regularization in actor's network - bias  
            :param kreg_l1_C :                  (float) Weight of L1 regularization in critic's network - kernel  
            :param kreg_l2_C :                  (float) Weight of L2 regularization in critic's network - kernel  
            :param breg_l1_C :                  (float) Weight of L1 regularization in critic's network - bias  
            :param breg_l2_C :                  (float) Weight of L2 regularization in critic's network - bias  
            :param u_max :                      (float array) Action upper bound array
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param NORMALIZE_INPUTS :           (bool) Flag to normalize inputs (state)
            :param state_norm_array :           (float array) Array used to normalize states
            :param MC :                         (bool) Flag to use MC or TD(n)
            :param cost_weights_terminal :      (float array) Running cost weights vector
            :param cost_weights_running :       (float array) Terminal cost weights vector 
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param dt :                         (float) Timestep

        :input w_S :                            (float) Sobolev-training weight
    '''

        self.env = env
        self.conf = conf

        self.w_S = w_S

        self.MSE = tf.keras.losses.MeanSquaredError()

        return
    
    def create_actor(self):
        ''' Create actor NN '''
        inputs = layers.Input(shape=(self.conf.nb_state,))
        
        lay1 = layers.Dense(self.conf.NH1,kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l1_A,self.conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.breg_l1_A,self.conf.breg_l2_A))(inputs)                                        
        leakyrelu1 = layers.LeakyReLU()(lay1)
        lay2 = layers.Dense(self.conf.NH2, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l1_A,self.conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.breg_l1_A,self.conf.breg_l2_A))(leakyrelu1)                                           
        leakyrelu2 = layers.LeakyReLU()(lay2)
        outputs = layers.Dense(self.conf.nb_action, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l1_A,self.conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.breg_l1_A,self.conf.breg_l2_A))(leakyrelu2) 

        model = tf.keras.Model(inputs, outputs)

        return model

    def create_critic_elu(self): 
        ''' Create critic NN - elu'''
        state_input = layers.Input(shape=(self.conf.nb_state,))

        state_out1 = layers.Dense(16, activation='elu')(state_input) 
        state_out2 = layers.Dense(32, activation='elu')(state_out1) 
        out_lay1 = layers.Dense(256, activation='elu')(state_out2)
        out_lay2 = layers.Dense(256, activation='elu')(out_lay1)
        
        outputs = layers.Dense(1)(out_lay2)

        model = tf.keras.Model([state_input], outputs)

        return model   
    
    def create_critic_sine_elu(self): 
        ''' Create critic NN - elu'''
        state_input = layers.Input(shape=(self.conf.nb_state,))

        state_out1 = SinusodialRepresentationDense(64, activation='sine')(state_input) 
        state_out2 = layers.Dense(64, activation='elu')(state_out1) 
        out_lay1 = SinusodialRepresentationDense(128, activation='sine')(state_out2)
        out_lay2 = layers.Dense(128, activation='elu')(out_lay1)

        outputs = layers.Dense(1)(out_lay2)

        model = tf.keras.Model([state_input], outputs)

        return model  
    
    def create_critic_sine(self): 
        ''' Create critic NN - elu'''
        state_input = layers.Input(shape=(self.conf.nb_state,))
        
        state_out1 = SinusodialRepresentationDense(64, activation='sine')(state_input) 
        state_out2 = SinusodialRepresentationDense(64, activation='sine')(state_out1) 
        out_lay1 = SinusodialRepresentationDense(128, activation='sine')(state_out2)
        out_lay2 = SinusodialRepresentationDense(128, activation='sine')(out_lay1)
        
        outputs = layers.Dense(1)(out_lay2)

        model = tf.keras.Model([state_input], outputs)

        return model   
    
    def create_critic_relu(self): 
        ''' Create critic NN - relu'''
        state_input = layers.Input(shape=(self.conf.nb_state,))

        state_out1 = layers.Dense(16, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(state_input) 
        leakyrelu1 = layers.LeakyReLU()(state_out1)
        
        state_out2 = layers.Dense(32, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(leakyrelu1) 
        leakyrelu2 = layers.LeakyReLU()(state_out2)
        out_lay1 = layers.Dense(self.conf.NH1, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(leakyrelu2)
        leakyrelu3 = layers.LeakyReLU()(out_lay1)
        out_lay2 = layers.Dense(self.conf.NH2, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(leakyrelu3)
        leakyrelu4 = layers.LeakyReLU()(out_lay2)
        
        outputs = layers.Dense(1, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(leakyrelu4)

        model = tf.keras.Model([state_input], outputs)

        return model     

    def eval(self, NN, input):
        ''' Compute the output of a NN given an input '''
        if not tf.is_tensor(input):
            input = tf.convert_to_tensor(input, dtype=tf.float32)

        if self.conf.NORMALIZE_INPUTS:
            input = normalize_tensor(input, self.conf.state_norm_arr)

        return NN(input, training=True)
    
    def custom_logarithm(self,input):
        # Calculate the logarithms based on the non-zero condition
        positive_log = tf.math.log(tf.math.maximum(input, 1e-7) + 1)
        negative_log = -tf.math.log(tf.math.maximum(-input, 1e-7) + 1)

        # Use the appropriate logarithm based on the condition
        result = tf.where(input > 0, positive_log, negative_log)

        return result    
    
    def compute_critic_grad(self, critic_model, target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch):
        ''' Compute the gradient of the critic NN '''
        #print(f'{state_batch.shape}, {state_next_rollout_batch.shape}, {partial_reward_to_go_batch.shape}, {dVdx_batch.shape}, {d_batch.shape}, {weights_batch.shape}')
        with tf.GradientTape() as tape: 
            # Compute value function tail if TD(n) is used
            if self.conf.MC:
                reward_to_go_batch = partial_reward_to_go_batch
            else:     
                target_values = self.eval(target_critic, state_next_rollout_batch)                                 # Compute Value at next state after conf.nsteps_TD_N steps given by target critic                 
                reward_to_go_batch = partial_reward_to_go_batch + (1-d_batch)*target_values                        # Compute batch of 1-step targets for the critic loss                    
            
            # Compute critic loss
            if self.w_S != 0:
                with tf.GradientTape() as tape2:
                    tape2.watch(state_batch)                  
                    critic_value = self.eval(critic_model, state_batch)   
                der_critic_value = tape2.gradient(critic_value, state_batch)
                
                critic_loss_v = self.MSE(reward_to_go_batch, critic_value, sample_weight=weights_batch)
                critic_loss_der = self.MSE(self.custom_logarithm(dVdx_batch[:,:-1]), self.custom_logarithm(der_critic_value[:,:-1]), sample_weight=weights_batch) # dV/dt not computed and so not used in the update
                
                critic_loss = critic_loss_der + self.w_S*critic_loss_v
            else:
                critic_value = self.eval(critic_model, state_batch)
                critic_loss = self.MSE(reward_to_go_batch, critic_value, sample_weight=weights_batch)

        # Compute the gradients of the critic loss w.r.t. critic's parameters
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)   

        return critic_grad, reward_to_go_batch, critic_value, self.eval(target_critic, state_batch)

    def compute_actor_grad(self, actor_model, critic_model, state_batch, term_batch, batch_size):
        ''' Compute the gradient of the actor NN '''
        print(type(state_batch))
        print(type(term_batch))
        #print(batch_size)
        if batch_size == None:
            batch_size = self.conf.BATCH_SIZE

        actions = self.eval(actor_model, state_batch)

        # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
        state_next_tf, ds_next_da = self.env.simulate_batch(state_batch.numpy(), actions.numpy()) , self.env.derivative_batch(state_batch.numpy(), actions.numpy())

        with tf.GradientTape() as tape:
            tape.watch(state_next_tf)
            critic_value_next = self.eval(critic_model,state_next_tf) 

        # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)                                           
        dV_ds_next = tape.gradient(critic_value_next, state_next_tf)

        cost_weights_terminal_reshaped = np.reshape(self.conf.cost_weights_terminal,[1,len(self.conf.cost_weights_terminal)])
        cost_weights_running_reshaped = np.reshape(self.conf.cost_weights_running,[1,len(self.conf.cost_weights_running)])
        with tf.GradientTape() as tape1:
            tape1.watch(actions)
            rewards_tf = self.env.reward_batch(term_batch.dot(cost_weights_terminal_reshaped) + (1-term_batch).dot(cost_weights_running_reshaped), state_batch.numpy(), actions)

        # dr_da = gradient of reward r(s,a) w.r.t. policy's action a
        dr_da = tape1.gradient(rewards_tf, actions, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        dr_da_reshaped = tf.reshape(dr_da, (batch_size, 1, self.conf.nb_action))
        
        # dr_ds' + dV_ds' (note: dr_ds' = 0)
        dQ_ds_next = tf.reshape(dV_ds_next, (batch_size, 1, self.conf.nb_state))        
        
        # (dr_ds' + dV_ds')*ds'_da
        dQ_ds_next_da = tf.matmul(dQ_ds_next, ds_next_da)
        
        # (dr_ds' + dV_ds')*ds'_da + dr_da
        dQ_da = dQ_ds_next_da + dr_da_reshaped

        # Now let's multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a 
        # and then let's autodifferentiate w.r.t theta_A (actor NN's parameters) to finally get -dQ/dtheta_A 
        with tf.GradientTape() as tape:
            tape.watch(actor_model.trainable_variables)
            actions = self.eval(actor_model, state_batch)
            
            actions_reshaped = tf.reshape(actions,(batch_size,self.conf.nb_action,1))
            dQ_da_reshaped = tf.reshape(dQ_da,(batch_size,1,self.conf.nb_action))    
            Q_neg = tf.matmul(-dQ_da_reshaped,actions_reshaped) 
            
            # Also here we need a scalar so we compute the mean -Q across the batch
            mean_Qneg = tf.math.reduce_mean(Q_neg)

        # Gradients of the actor loss w.r.t. actor's parameters
        actor_grad = tape.gradient(mean_Qneg, actor_model.trainable_variables)

        return actor_grad

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
    print(mean_Qneg)
    actor_grad = torch.autograd.grad(mean_Qneg, actor_model.parameters())
    #actor_model.zero_grad()
    #mean_Qneg.backward()
    #actor_grad = [param.grad for param in actor_model.parameters()]

    return actor_grad

class RL_AC_tf:
    def __init__(self, env, NN, conf, N_try):
        '''    
        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)

            :parma critic_type :                (str) Activation function to use for the critic NN
            :param LR_SCHEDULE :                (bool) Flag to use a scheduler for the learning rates
            :param boundaries_schedule_LR_C :   (list) Boudaries of critic LR
            :param values_schedule_LR_C :       (list) Values of critic LR
            :param boundaries_schedule_LR_A :   (list) Boudaries of actor LR
            :param values_schedule_LR_A :       (list) Values of actor LR
            :param CRITIC_LEARNING_RATE :       (float) Learning rate for the critic network
            :param ACTOR_LEARNING_RATE :        (float) Learning rate for the policy network
            :param fresh_factor :               (float) Refresh factor
            :param prioritized_replay_alpha :   (float) α determines how much prioritization is used
            :param prioritized_replay_eps :     (float) It's a small positive constant that prevents the edge-case of transitions not being revisited once their error is zero
            :param UPDATE_LOOPS :               (int array) Number of updates of both critic and actor performed every EP_UPDATE episodes
            :param save_interval :              (int) save NNs interval
            :param env_RL :                     (bool) Flag RL environment
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param MC :                         (bool) Flag to use MC or TD(n)
            :param nsteps_TD_N :                (int) Number of lookahed steps if TD(n) is used
            :param UPDATE_RATE :                (float) Homotopy rate to update the target critic network if TD(n) is used
            :param cost_weights_terminal :      (float array) Running cost weights vector
            :param cost_weights_running :       (float array) Terminal cost weights vector 
            :param dt :                         (float) Timestep
            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param NNs_path :                   (str) NNs path
            :param NSTEPS :                     (int) Max episode length

    '''
        self.env = env
        self.NN = NN
        self.conf = conf

        self.N_try = N_try

        self.actor_model = None
        self.critic_model = None
        self.target_critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.init_rand_state = None
        self.NSTEPS_SH = 0
        self.control_arr = None
        self.state_arr = None
        self.ee_pos_arr = None
        self.exp_counter = np.zeros(conf.REPLAY_SIZE)

        return
    
    def setup_model(self, recover_training=None):
        ''' Setup RL model '''
        # Create actor, critic and target NNs
        self.actor_model = self.NN.create_actor()

        if self.conf.critic_type == 'elu':
            self.critic_model = self.NN.create_critic_elu()
            self.target_critic = self.NN.create_critic_elu()
        elif self.conf.critic_type == 'sine':
            self.critic_model = self.NN.create_critic_sine()
            self.target_critic = self.NN.create_critic_sine()
        elif self.conf.critic_type == 'sine-elu':
            self.critic_model = self.NN.create_critic_sine_elu()
            self.target_critic = self.NN.create_critic_sine_elu()
        else:
            self.critic_model = self.NN.create_critic_relu()
            self.target_critic = self.NN.create_critic_relu()

        # Set optimizer specifying the learning rates
        if self.conf.LR_SCHEDULE:
            # Piecewise constant decay schedule
            self.CRITIC_LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.conf.boundaries_schedule_LR_C, self.conf.values_schedule_LR_C) 
            self.ACTOR_LR_SCHEDULE  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.conf.boundaries_schedule_LR_A, self.conf.values_schedule_LR_A)
            self.critic_optimizer   = tf.keras.optimizers.Adam(self.CRITIC_LR_SCHEDULE)
            self.actor_optimizer    = tf.keras.optimizers.Adam(self.ACTOR_LR_SCHEDULE)
        else:
            self.critic_optimizer   = tf.keras.optimizers.Adam(self.conf.CRITIC_LEARNING_RATE)
            self.actor_optimizer    = tf.keras.optimizers.Adam(self.conf.ACTOR_LEARNING_RATE)

        # Set initial weights of the NNs
        if recover_training is not None: 
            NNs_path_rec = str(recover_training[0])
            N_try = recover_training[1]
            update_step_counter = recover_training[2]
            self.actor_model.load_weights("{}/N_try_{}/actor_{}.h5".format(NNs_path_rec,N_try,update_step_counter))
            self.critic_model.load_weights("{}/N_try_{}/critic_{}.h5".format(NNs_path_rec,N_try,update_step_counter))
            self.target_critic.load_weights("{}/N_try_{}/target_critic_{}.h5".format(NNs_path_rec,N_try,update_step_counter))
        else:
            self.target_critic.set_weights(self.critic_model.get_weights())   

    def update(self, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, term_batch, weights_batch, batch_size=None):
        ''' Update both critic and actor '''
        # Update the critic backpropagating the gradients
        critic_grad, reward_to_go_batch, critic_value, target_critic_value = self.NN.compute_critic_grad(self.critic_model, self.target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Update the actor backpropagating the gradients
        actor_grad = self.NN.compute_actor_grad(self.actor_model, self.critic_model, state_batch, term_batch, batch_size)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        return reward_to_go_batch, critic_value, target_critic_value
    
    @tf.function
    def update_target(self, target_weights, weights):
        ''' Update target critic NN '''
        tau = self.conf.UPDATE_RATE
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def learn_and_update(self, update_step_counter, buffer, ep):
        ''' Sample experience and update buffer priorities and NNs '''
        for i in range(int(self.conf.UPDATE_LOOPS[ep])):
            # Sample batch of transitions from the buffer
            state_batch, partial_reward_to_go_batch, state_next_rollout_batch, dVdx_batch, d_batch, term_batch, weights_batch, batch_idxes = buffer.sample()

            # Update both critic and actor
            reward_to_go_batch, critic_value, target_critic_value = self.update(state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, term_batch, weights_batch)

            # Update buffer priorities
            if self.conf.prioritized_replay_alpha != 0:                                
                buffer.update_priorities(batch_idxes, reward_to_go_batch, critic_value, target_critic_value)  

            # Update target critic
            if not self.conf.MC:
                self.update_target(self.target_critic.variables, self.critic_model.variables)

            update_step_counter += 1

            # Plot rollouts and save the NNs every conf.log_rollout_interval-training episodes
            if update_step_counter%self.conf.save_interval == 0:
                self.RL_save_weights(update_step_counter)

        return update_step_counter
    
    def RL_Solve(self, TO_controls, TO_states, TO_step_cost):
        ''' Solve RL problem '''
        ep_return = 0                                                                 # Initialize the return
        rwrd_arr = np.empty(self.NSTEPS_SH+1)                                         # Reward array
        state_next_rollout_arr = np.zeros((self.NSTEPS_SH+1, self.conf.nb_state))     # Next state array
        partial_reward_to_go_arr = np.empty(self.NSTEPS_SH+1)                         # Partial cost-to-go array
        total_reward_to_go_arr = np.empty(self.NSTEPS_SH+1)                           # Total cost-to-go array
        term_arr = np.zeros(self.NSTEPS_SH+1)                                         # Episode-termination flag array
        term_arr[-1] = 1
        done_arr = np.zeros(self.NSTEPS_SH+1)                                         # Episode-MC-termination flag array

        # START RL EPISODE
        self.control_arr = TO_controls # action clipped in TO
        
        if self.conf.env_RL:
            for step_counter in range(self.NSTEPS_SH):
                # Simulate actions and retrieve next state and compute reward
                self.state_arr[step_counter+1,:], rwrd_arr[step_counter] = self.env.step(self.conf.cost_weights_running, self.state_arr[step_counter,:], self.control_arr[step_counter,:])

                # Compute end-effector position
                self.ee_pos_arr[step_counter+1,:] = self.env.get_end_effector_position(self.state_arr[step_counter+1, :])
            rwrd_arr[-1] = self.env.reward(self.conf.cost_weights_terminal, self.state_arr[-1,:])
        else:
            self.state_arr, rwrd_arr = TO_states, -TO_step_cost

        ep_return = sum(rwrd_arr)

        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        for i in range(self.NSTEPS_SH+1):
            # set final lookahead step depending on whether Monte Cartlo or TD(n) is used
            if self.conf.MC:
                final_lookahead_step = self.NSTEPS_SH
                done_arr[i] = 1 
            else:
                final_lookahead_step = min(i+self.conf.nsteps_TD_N, self.NSTEPS_SH)
                if final_lookahead_step == self.NSTEPS_SH:
                    done_arr[i] = 1 
                else:
                    state_next_rollout_arr[i,:] = self.state_arr[final_lookahead_step+1,:]
            
            # Compute the partial and total cost to go
            partial_reward_to_go_arr[i] = np.float32(sum(rwrd_arr[i:final_lookahead_step+1]))
            total_reward_to_go_arr[i] = np.float32(sum(rwrd_arr[i:self.NSTEPS_SH+1]))

        return self.state_arr, partial_reward_to_go_arr, total_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, self.ee_pos_arr
    
    def RL_save_weights(self, update_step_counter='final'):
        ''' Save NN weights '''
        self.actor_model.save_weights(self.conf.NNs_path+"/N_try_{}/actor_{}.h5".format(self.N_try,update_step_counter))
        self.critic_model.save_weights(self.conf.NNs_path+"/N_try_{}/critic_{}.h5".format(self.N_try,update_step_counter))
        self.target_critic.save_weights(self.conf.NNs_path+"/N_try_{}/target_critic_{}.h5".format(self.N_try,update_step_counter))

    def create_TO_init(self, ep, ICS):
        ''' Create initial state and initial controls for TO '''
        self.init_rand_state = ICS    
        
        self.NSTEPS_SH = self.conf.NSTEPS - int(self.init_rand_state[-1]/self.conf.dt)
        if self.NSTEPS_SH == 0:
            return None, None, None, None, 0

        # Initialize array to store RL state, control, and end-effector trajectories
        self.control_arr = np.empty((self.NSTEPS_SH, self.conf.nb_action))
        self.state_arr = np.empty((self.NSTEPS_SH+1, self.conf.nb_state))
        self.ee_pos_arr = np.empty((self.NSTEPS_SH+1,3))

        # Set initial state and end-effector position
        self.state_arr[0,:] = self.init_rand_state
        self.ee_pos_arr[0,:] = self.env.get_end_effector_position(self.state_arr[0, :])

        # Initialize array to initialize TO state and control variables
        init_TO_controls = np.zeros((self.NSTEPS_SH, self.conf.nb_action))
        init_TO_states = np.zeros(( self.NSTEPS_SH+1, self.conf.nb_state))

        # Set initial state 
        init_TO_states[0,:] = self.init_rand_state

        # Simulate actor's actions to compute the state trajectory used to initialize TO state variables (use ICS for state and 0 for control if it is the first episode otherwise use policy rollout)
        success_init_flag = 1
        for i in range(self.NSTEPS_SH):   
            if ep == 0:
                init_TO_controls[i,:] = np.zeros(self.conf.nb_action)
            else:
                init_TO_controls[i,:] = tf.squeeze(self.NN.eval(self.actor_model, np.array([init_TO_states[i,:]]))).numpy()
            init_TO_states[i+1,:] = self.env.simulate(init_TO_states[i,:],init_TO_controls[i,:])
            if np.isnan(init_TO_states[i+1,:]).any():
                success_init_flag = 0
                return None, None, None, None, success_init_flag

        return self.init_rand_state, init_TO_states, init_TO_controls, self.NSTEPS_SH, success_init_flag


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


env = Environment(conf)                                                                                 # Create environment instances
envtf = DoubleIntegratortf(conf)
env_TO = Environment_TO
NNtorch = NN(env, conf, w_S)                                                                            # Create NN instance
NNtf = NN_tf(envtf, conf, w_S)
TrOptorch = TO_Casadi(env, conf, env_TO, w_S)
TrOptf = TO_Casadi(envtf, conf, env_TO, w_S)                                                                # Create TO instance
RLACtorch = RL_AC(env, NNtorch, conf, 0)                                                                     # Create RL instance
RLACtf = RL_AC_tf(envtf, NNtf, conf, 0)
buffer = ReplayBuffer(conf) if conf.prioritized_replay_alpha == 0 else PrioritizedReplayBuffer(conf)    # Create an empty (prioritized) replay buffer

# Set initial weights of the NNs, initialize the counter of the updates and setup NN models

update_step_counter = 0

RLACtorch.setup_model()
RLACtf.setup_model()

# Save initial weights of the NNs
RLACtorch.RL_save_weights(update_step_counter)
RLACtf.RL_save_weights(update_step_counter)

# Initialize arrays to store the reward history of each episode and the average reward history of last 100 episodes
ep_arr_idx = 0
ep_reward_arr = np.zeros(conf.NEPISODES-ep_arr_idx)*np.nan                                                                                     

def compute_sample(args, rein, to):
    ''' Create samples solving TO problems starting from given ICS '''
    ep = args[0]
    ICS = args[1]

    # Create initial TO #
    init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH, success_init_flag = rein.create_TO_init(ep, ICS)
    if success_init_flag == 0:
        return None
        
    # Solve TO problem #
    TO_controls, TO_states, success_flag, TO_ee_pos_arr, TO_step_cost, dVdx = to.TO_Solve(init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)
    if success_flag == 0:
        return None
    
    # Collect experiences 
    state_arr, partial_reward_to_go_arr, total_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr  = rein.RL_Solve(TO_controls, TO_states, TO_step_cost)

    if conf.env_RL == 0:
        RL_ee_pos_arr = TO_ee_pos_arr

    return NSTEPS_SH, TO_controls, TO_ee_pos_arr, dVdx, state_arr.tolist(), partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr

def create_unif_TO_init(e, n_UICS=1):
    ''' Create n uniformely distributed ICS '''
    # Create ICS TO #
    init_rand_state = e.reset()
    
    return init_rand_state

time_start = time.time()

for ep in range(conf.NLOOPS): 
    # Generate and store conf.EP_UPDATE random-uniform ICS
    tmp_torch = []
    tmp_tf = []
    init_rand_state = []
    for i in range(conf.EP_UPDATE):
        init_rand_state.append(create_unif_TO_init(i))

    for i in range(conf.EP_UPDATE):
        result_torch = compute_sample((ep, init_rand_state[i]), RLACtorch, TrOptorch)
        result_tf    = compute_sample((ep, init_rand_state[i]), RLACtf   , TrOptf   )
        tmp_torch.append(result_torch)
        tmp_tf.append(result_tf)

    # Remove unsuccessful TO problems and update EP_UPDATE
    tmp_torch = [x for x in tmp_torch if x is not None]
    tmp_tf = [x for x in tmp_tf if x is not None]
    NSTEPS_SH, TO_controls, ee_pos_arr_TO, dVdx, state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, ee_pos_arr_RL = zip(*tmp)

    # Update the buffer
    buffer.add(state_arr, partial_reward_to_go_arr, state_next_rollout_arr, dVdx, done_arr, term_arr)

    # Update NNs
    update_step_counter = RLAC.learn_and_update(update_step_counter, buffer, ep)

    # plot Critic value function
    #plot_fun.plot_Critic_Value_function(RLAC.critic_model, update_step_counter, system_id) ###

    # Plot rollouts and state and control trajectories
    if update_step_counter%conf.plot_rollout_interval_diff_loc == 0 or system_id == 'single_integrator' or system_id == 'double_integrator' or system_id == 'car_park' or system_id == 'car' or system_id == 'manipulator':
        print("System: {} - N_try = {}".format(conf.system_id, N_try))
        plot_fun.plot_Critic_Value_function(RLAC.critic_model, update_step_counter, system_id)
        plot_fun.plot_traj_from_ICS(np.array(conf.init_states_sim), TrOp, RLAC, update_step_counter=update_step_counter, ep=ep,steps=conf.NSTEPS, init=1)

    # Update arrays to store the reward history and its average
    ep_reward_arr[ep_arr_idx:ep_arr_idx+len(tmp)] = ep_return
    ep_arr_idx += len(tmp)

    for i in range(len(tmp)):
        print("Episode  {}  --->   Return = {}".format(ep*len(tmp) + i, ep_return[i]))

    if update_step_counter > conf.NUPDATES:
        break

time_end = time.time()
print('Elapsed time: ', time_end-time_start)

if conf.profile:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

# Plot returns
plot_fun.plot_Return(ep_reward_arr)

# Save networks at the end of the training
RLAC.RL_save_weights()

# Simulate the final policy
plot_fun.rollout(update_step_counter, RLAC.actor_model, conf.init_states_sim)
