#NOTE: This is a very messy script. To debug please collapse all classes 
# and focus on the driver code inside the 'main' if statement at the end
# This script also focuses on testing for DOUBLE INTEGRSTOR only. To
# run it pase the following command in cli:
# python3 testing_fullpass.py --system-id='double_integrator' --seed=0 --nb-cpus=15 --w-S=1e-2 --test-n=0
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import unittest
import numpy as np
import tensorflow as tf
import torch
import uuid
import math
import importlib
import torch.nn as nn
from utils import normalize_tensor
from TO import TO_Casadi
import random
from keras import layers, regularizers, initializers
from tf_siren import SinusodialRepresentationDense
import pinocchio as pin
import argparse
import shutil
from plot_utils import PLOT
from segment_tree import SumSegmentTree, MinSegmentTree
from siren_pytorch import Siren
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d
import time
from replay_buffer import ReplayBuffer

def normalize_tensor_tf(state, state_norm_arr):
    ''' Retrieve state from normalized state - tensor '''
    state_norm_time = tf.concat([tf.zeros([state.shape[0], state.shape[1]-1]), tf.reshape(((state[:,-1]) / state_norm_arr[-1])*2 - 1,[state.shape[0],1])],1)
    state_norm_no_time = state / state_norm_arr
    mask = tf.concat([tf.ones([state.shape[0], state.shape[1]-1]), tf.zeros([state.shape[0], 1])],1)
    state_norm = state_norm_no_time * mask + state_norm_time * (1 - mask)

    return state_norm

class ReplayBuffer_tf(object):
    def __init__(self, conf):
        '''
        :input conf :                           (Configuration file)

            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param nb_state :                   (int) State size (robot state size + 1)
        '''

        self.conf = conf
        self.storage_mat = np.zeros((conf.REPLAY_SIZE, conf.nb_state + 1 + conf.nb_state + conf.nb_state + 1 + 1))
        self.next_idx = 0
        self.full = 0
        self.exp_counter = np.zeros(conf.REPLAY_SIZE)

    def add(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
        ''' Add transitions to the buffer '''
        data = self.concatenate_sample(obses_t, rewards, obses_t1, dVdxs, dones, terms)

        if len(data) + self.next_idx > self.conf.REPLAY_SIZE:
            self.storage_mat[self.next_idx:,:] = data[:self.conf.REPLAY_SIZE-self.next_idx,:]
            self.storage_mat[:self.next_idx+len(data)-self.conf.REPLAY_SIZE,:] = data[self.conf.REPLAY_SIZE-self.next_idx:,:]
            self.full = 1
        else:
            self.storage_mat[self.next_idx:self.next_idx+len(data),:] = data

        self.next_idx = (self.next_idx + len(data)) % self.conf.REPLAY_SIZE

    def sample(self):
        ''' Sample a batch of transitions '''
        # Select indexes of the batch elements
        if self.full:
            max_idx = self.conf.REPLAY_SIZE
        else:
            max_idx = self.next_idx
        idxes = np.random.randint(0, max_idx, size=self.conf.BATCH_SIZE) 

        obses_t = self.storage_mat[idxes, :self.conf.nb_state]
        rewards = self.storage_mat[idxes, self.conf.nb_state:self.conf.nb_state+1]
        obses_t1 = self.storage_mat[idxes, self.conf.nb_state+1:self.conf.nb_state*2+1]
        dVdxs = self.storage_mat[idxes, self.conf.nb_state*2+1:self.conf.nb_state*3+1]
        dones = self.storage_mat[idxes, self.conf.nb_state*3+1:self.conf.nb_state*3+2]
        terms = self.storage_mat[idxes, self.conf.nb_state*3+2:self.conf.nb_state*3+3]

        # Priorities not used
        weights = np.ones((self.conf.BATCH_SIZE,1))
        batch_idxes = None

        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dVdxs, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dVdxs, dones, weights)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, batch_idxes

    def concatenate_sample(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
        ''' Convert batch of transitions into a tensor '''
        obses_t = np.concatenate(obses_t, axis=0)
        rewards = np.concatenate(rewards, axis=0)                                 
        obses_t1 = np.concatenate(obses_t1, axis=0)
        dVdxs = np.concatenate(dVdxs, axis=0)
        dones = np.concatenate(dones, axis=0)
        terms = np.concatenate(terms, axis=0)
        
        return np.concatenate((obses_t, rewards.reshape(-1,1), obses_t1, dVdxs, dones.reshape(-1,1), terms.reshape(-1,1)),axis=1)
    
    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dVdxs, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)                                  
        obses_t1 = tf.convert_to_tensor(obses_t1, dtype=tf.float32)
        dVdxs = tf.convert_to_tensor(dVdxs, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, weights

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

        if conf.NORMALIZE_INPUTS:
            input = normalize_tensor_tf(input, conf.state_norm_arr)

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
        with tf.GradientTape() as tape: 
            # Compute value function tail if TD(n) is used
            if conf.MC:
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
        if batch_size == None:
            batch_size = conf.BATCH_SIZE

        actions = self.eval(actor_model, state_batch)

        # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
        state_next_tf, ds_next_da = self.env.simulate_batch(state_batch.numpy(), actions.numpy()) , self.env.derivative_batch(state_batch.numpy(), actions.numpy())

        with tf.GradientTape() as tape:
            tape.watch(state_next_tf)
            critic_value_next = self.eval(critic_model,state_next_tf) 

        # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)                                           
        dV_ds_next = tape.gradient(critic_value_next, state_next_tf)

        cost_weights_terminal_reshaped = np.reshape(conf.cost_weights_terminal,[1,len(conf.cost_weights_terminal)])
        cost_weights_running_reshaped = np.reshape(conf.cost_weights_running,[1,len(conf.cost_weights_running)])
        with tf.GradientTape() as tape1:
            tape1.watch(actions)
            rewards_tf = self.env.reward_batch(term_batch.dot(cost_weights_terminal_reshaped) + (1-term_batch).dot(cost_weights_running_reshaped), state_batch.numpy(), actions)

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
            actions = self.eval(actor_model, state_batch)
            
            actions_reshaped = tf.reshape(actions,(batch_size,conf.nb_action,1))
            dQ_da_reshaped = tf.reshape(dQ_da,(batch_size,1,conf.nb_action))    
            Q_neg = tf.matmul(-dQ_da_reshaped,actions_reshaped) 
            
            # Also here we need a scalar so we compute the mean -Q across the batch
            mean_Qneg = tf.math.reduce_mean(Q_neg)

        # Gradients of the actor loss w.r.t. actor's parameters
        actor_grad = tape.gradient(mean_Qneg, actor_model.trainable_variables)

        return actor_grad

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

class NN_torch:
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
        self.MSE = WeightedMSELoss()
        return
    
    def create_actor(self, weights=None):
        ''' Create actor NN '''
        #Tested Successfully#
        model = nn.Sequential(
            nn.Linear(self.conf.nb_state, self.conf.NH1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(self.conf.NH1, self.conf.NH2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(self.conf.NH2, self.conf.nb_action)
        )
        if weights is not None:
            index = 0
            for layer in model:
                if isinstance(layer, nn.Linear):
                    # Extract the weight and bias arrays
                    weight_array = torch.t(torch.tensor(weights[index][0]))
                    bias_array = torch.t(torch.tensor(weights[index][1]))
                    
                    # Set weights
                    with torch.no_grad():
                        layer.weight.copy_(weight_array)
                        layer.bias.copy_(bias_array)
                        
                    # Move to the next set of weights
                    index += 1
                else:
                    continue
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.5)
                    nn.init.constant_(layer.bias, 0.5)
        return model

    def create_critic_elu(self, weights=None): 
        ''' Create critic NN - elu'''
        #Tested Successfully#
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
        if weights is not None:
            index = 0
            for layer in model:
                if isinstance(layer, nn.Linear):
                    # Extract the weight and bias arrays
                    weight_array = torch.t(torch.tensor(weights[index][0]))
                    bias_array = torch.t(torch.tensor(weights[index][1]))
                    
                    # Set weights
                    with torch.no_grad():
                        layer.weight.copy_(weight_array)
                        layer.bias.copy_(bias_array)
                        
                    # Move to the next set of weights
                    index += 1
                else:
                    continue
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.5)
                    nn.init.constant_(layer.bias, 0.5)
        return model
    
    def create_critic_sine_elu(self, weights=None): 
        ''' Create critic NN - elu'''
        #Tested Successfully#
        model = nn.Sequential(
            Siren(conf.nb_state, 64),
            nn.Linear(64, 64),
            nn.ELU(),
            Siren(64, 128),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128,1)
        )
        if weights is not None:
            index = 0
            for layer in model:
                if isinstance(layer, nn.Linear) or isinstance(layer, Siren):
                    # Extract the weight and bias arrays
                    weight_array = torch.t(torch.tensor(weights[index][0]))
                    bias_array = torch.t(torch.tensor(weights[index][1]))
                    
                    # Set weights
                    with torch.no_grad():
                        layer.weight.copy_(weight_array)
                        layer.bias.copy_(bias_array)
                        
                    # Move to the next set of weights
                    index += 1
                else:
                    continue
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.5)
                    nn.init.constant_(layer.bias, 0.5)
        return model
        
    def create_critic_sine(self, weights=None): 
        ''' Create critic NN - elu'''
        model = nn.Sequential(
            Siren(conf.nb_state, 64),
            Siren(64, 64),
            Siren(64, 128),
            Siren(128, 128),
            nn.Linear(128, 1)
        )
        if weights is not None:
            index = 0
            for layer in model:
                if isinstance(layer, nn.Linear) or isinstance(layer, Siren):
                    # Extract the weight and bias arrays
                    weight_array = torch.t(torch.tensor(weights[index][0]))
                    bias_array = torch.t(torch.tensor(weights[index][1]))
                    
                    # Set weights
                    with torch.no_grad():
                        layer.weight.copy_(weight_array)
                        layer.bias.copy_(bias_array)
                        
                    # Move to the next set of weights
                    index += 1
                else:
                    continue
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.5)
                    nn.init.constant_(layer.bias, 0.5)
        return model
        
    def create_critic_relu(weights=None): 
        ''' Create critic NN - relu'''
        #Tested Successfully#
        model = nn.Sequential(
            nn.Linear(conf.nb_state, 16),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(16, 32),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(32, conf.NH1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(conf.NH1, conf.NH2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(conf.NH2, 1)
        )
        if weights is not None:
            index = 0
            for layer in model:
                if isinstance(layer, nn.Linear) or isinstance(layer, Siren):
                    # Extract the weight and bias arrays
                    weight_array = torch.t(torch.tensor(weights[index][0]))
                    bias_array = torch.t(torch.tensor(weights[index][1]))
                    
                    # Set weights
                    with torch.no_grad():
                        layer.weight.copy_(weight_array)
                        layer.bias.copy_(bias_array)
                        
                    # Move to the next set of weights
                    index += 1
                else:
                    continue
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.5)
                    nn.init.constant_(layer.bias, 0.5)
        return model
    
    def eval(self, NN, input):
        ''' Compute the output of a NN given an input '''
        #Tested Successfully#
        if not torch.is_tensor(input):
            if isinstance(input, list):
                input = np.array(input)
            input = torch.tensor(input, dtype=torch.float32)

        if conf.NORMALIZE_INPUTS:
            input = normalize_tensor(input, torch.tensor(conf.state_norm_arr))

        return NN(input)
    
    def custom_logarithm(self,input):
        #Tested Successfully#
        # Calculate the logarithms based on the non-zero condition
        positive_log = torch.log(torch.maximum(input, torch.tensor(1e-7)) + 1)
        negative_log = -torch.log(torch.maximum(-input, torch.tensor(1e-7)) + 1)

        # Use the appropriate logarithm based on the condition
        result = torch.where(input > 0, positive_log, negative_log)

        return result    
    
    def compute_critic_grad(self, critic_model, target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch):
        ''' Compute the gradient of the critic NN. Does not return the critic gradients since 
        they will be present in .grad attributes of the critic_model after execution.'''
        #NOTE: regularization added#
        #Tested Successfully#
        reward_to_go_batch = partial_reward_to_go_batch if conf.MC else partial_reward_to_go_batch + (1 - d_batch) * self.eval(target_critic, state_next_rollout_batch)

        critic_model.zero_grad()
        if self.w_S != 0:
            state_batch.requires_grad_(True)
            critic_value = self.eval(critic_model, state_batch)
            der_critic_value = torch.autograd.grad(outputs=critic_value, inputs=state_batch, grad_outputs=torch.ones_like(critic_value), create_graph=True)[0]
            critic_loss_v = self.MSE(reward_to_go_batch, critic_value, weights=weights_batch)
            critic_loss_der = self.MSE(self.custom_logarithm(dVdx_batch[:,:-1]), self.custom_logarithm(der_critic_value[:,:-1]), weights=weights_batch) # dV/dt not computed and so not used in the update
            critic_loss = critic_loss_der + self.w_S*critic_loss_v
        else:
            critic_value = self.eval(critic_model, state_batch)
            critic_loss = self.MSE(reward_to_go_batch, critic_value, weights=weights_batch)
        
        #critic_grad = torch.autograd.grad(critic_loss, critic_model.parameters())
        total_loss = critic_loss #+ self.compute_reg_loss(critic_model, False)
        critic_model.zero_grad()
        total_loss.backward()

        return reward_to_go_batch, critic_value, self.eval(target_critic, state_batch)
        #return critic_grad, reward_to_go_batch, critic_value, target_critic(state_batch)

    def compute_actor_grad(self, actor_model, critic_model, state_batch, term_batch, batch_size):
        ''' 
        Compute and apply the gradient of the actor NN. Does not return anything since the 
        gradients will be present in .grad attributes of the actor_model after execution.
        '''
        #NOTE: This function was tested against its tensorflow equivalent, and it produced 
        #gradients exactly 10 times larger. All intermediate values are the same. Requires
        #further investigation (could be the cause of the large errors). Also, regularization
        #was added
        actor_model.zero_grad()
        if batch_size is None:
            batch_size = conf.BATCH_SIZE

        actions = self.eval(actor_model, state_batch)

        # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
        act_np = actions.detach().numpy()
        state_next_tf, ds_next_da = self.env.simulate_batch(state_batch.detach().numpy(), act_np), self.env.derivative_batch(state_batch.detach().numpy(), act_np)
        
        state_next_tf = state_next_tf.clone().detach().to(dtype=torch.float32).requires_grad_(True)
        ds_next_da = ds_next_da.clone().detach().to(dtype=torch.float32).requires_grad_(True)

        # Compute critic value at the next state
        critic_value_next = self.eval(critic_model, state_next_tf)

        # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)
        dV_ds_next = torch.autograd.grad(outputs=critic_value_next, inputs=state_next_tf,
                                        grad_outputs=torch.ones_like(critic_value_next),
                                        create_graph=True)[0]

        cost_weights_terminal_reshaped = torch.tensor(conf.cost_weights_terminal, dtype=torch.float32).reshape(1, -1)
        cost_weights_running_reshaped = torch.tensor(conf.cost_weights_running, dtype=torch.float32).reshape(1, -1)

        # Compute rewards
        rewards_tf = self.env.reward_batch(term_batch.dot(cost_weights_terminal_reshaped) + (1-term_batch).dot(cost_weights_running_reshaped), state_batch.detach().numpy(), actions)

        # dr_da = gradient of reward r(s,a) w.r.t. policy's action a
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
        actions = self.eval(actor_model, state_batch)
        actions_reshaped = actions.view(batch_size, conf.nb_action, 1)
        dQ_da_reshaped = dQ_da.view(batch_size, 1, conf.nb_action)
        #Q_neg = torch.bmm(-dQ_da_reshaped, actions_reshaped)
        Q_neg = torch.matmul(-dQ_da_reshaped, actions_reshaped)

        # Compute the mean -Q across the batch
        mean_Qneg = Q_neg.mean()
        total_loss = mean_Qneg #+ self.compute_reg_loss(actor_model, True)

        # Gradients of the actor loss w.r.t. actor's parameters
        actor_model.zero_grad()
        #actor_grad = torch.autograd.grad(mean_Qneg, actor_model.parameters())
        total_loss.backward()
        for param in actor_model.parameters():
            if param.grad is not None:
                param.grad.data /= 10
        #actor_grad = [param.grad for param in actor_model.parameters()]
        #print()
        #return actor_grad
        return

    def compute_reg_loss(self, model, actor):
        '''Computes L1 and L2 regularization losses for weights and biases'''
        #NOTE: layers in the original tf code were using kreg_l2_C (from conf) for all regularization parameters. 
        #This doesn't make sense and was changed here. Also, the original codebase used the keras 
        #bias_regularizer and kernel_regularizer variables, but never accessed the actor_model.losses
        #parameter to actually use the regularization loss in gradient computations
        reg_loss = 0
        kreg_l1 = 0
        kreg_l2 = 0
        breg_l1 = 0
        breg_l2 = 0
        if actor:
            kreg_l1 = conf.kreg_l1_A
            kreg_l2 = conf.kreg_l2_A
            breg_l1 = conf.breg_l1_A
            breg_l2 = conf.breg_l2_A
        else:
            kreg_l1 = conf.kreg_l1_C
            kreg_l2 = conf.kreg_l2_C
            breg_l1 = conf.breg_l1_C
            breg_l2 = conf.breg_l2_C

        for layer in model:
            if isinstance(layer, nn.Linear):
                if kreg_l1 > 0:
                    l1_regularization_w = kreg_l1 * torch.sum(torch.abs(layer.weight))
                    reg_loss += l1_regularization_w
                if kreg_l2 > 0:
                    l2_regularization_w = kreg_l2 * torch.sum(torch.pow(layer.weight, 2))
                    reg_loss += l2_regularization_w

                # Regularization for biases
                if breg_l1 > 0:
                    l1_regularization_b = breg_l1 * torch.sum(torch.abs(layer.bias))
                    reg_loss += l1_regularization_b
                if breg_l2 > 0:
                    l2_regularization_b = breg_l2 * torch.sum(torch.pow(layer.bias, 2))
                    reg_loss += l2_regularization_b
        return reg_loss

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
        self.critic_model = self.NN.create_critic_sine()
        self.target_critic = self.NN.create_critic_sine()

        # Set optimizer specifying the learning rates
        if conf.LR_SCHEDULE:
            # Piecewise constant decay schedule
            self.CRITIC_LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(conf.boundaries_schedule_LR_C, conf.values_schedule_LR_C) 
            self.ACTOR_LR_SCHEDULE  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(conf.boundaries_schedule_LR_A, conf.values_schedule_LR_A)
            self.critic_optimizer   = tf.keras.optimizers.Adam(self.CRITIC_LR_SCHEDULE)
            self.actor_optimizer    = tf.keras.optimizers.Adam(self.ACTOR_LR_SCHEDULE)
        else:
            self.critic_optimizer   = tf.keras.optimizers.Adam(conf.CRITIC_LEARNING_RATE)
            self.actor_optimizer    = tf.keras.optimizers.Adam(conf.ACTOR_LEARNING_RATE)

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
    
    #@tf.function
    def update_target(self, target_weights, weights):
        ''' Update target critic NN '''
        tau = conf.UPDATE_RATE
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
        state_next_rollout_arr = np.zeros((self.NSTEPS_SH+1, conf.nb_state))     # Next state array
        partial_reward_to_go_arr = np.empty(self.NSTEPS_SH+1)                         # Partial cost-to-go array
        total_reward_to_go_arr = np.empty(self.NSTEPS_SH+1)                           # Total cost-to-go array
        term_arr = np.zeros(self.NSTEPS_SH+1)                                         # Episode-termination flag array
        term_arr[-1] = 1
        done_arr = np.zeros(self.NSTEPS_SH+1)                                         # Episode-MC-termination flag array

        # START RL EPISODE
        self.control_arr = TO_controls # action clipped in TO
        
        if conf.env_RL:
            for step_counter in range(self.NSTEPS_SH):
                # Simulate actions and retrieve next state and compute reward
                self.state_arr[step_counter+1,:], rwrd_arr[step_counter] = self.env.step(conf.cost_weights_running, self.state_arr[step_counter,:], self.control_arr[step_counter,:])

                # Compute end-effector position
                self.ee_pos_arr[step_counter+1,:] = self.env.get_end_effector_position(self.state_arr[step_counter+1, :])
            rwrd_arr[-1] = self.env.reward(conf.cost_weights_terminal, self.state_arr[-1,:])
        else:
            self.state_arr, rwrd_arr = TO_states, -TO_step_cost

        ep_return = sum(rwrd_arr)

        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        for i in range(self.NSTEPS_SH+1):
            # set final lookahead step depending on whether Monte Cartlo or TD(n) is used
            if conf.MC:
                final_lookahead_step = self.NSTEPS_SH
                done_arr[i] = 1 
            else:
                final_lookahead_step = min(i+conf.nsteps_TD_N, self.NSTEPS_SH)
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
        self.actor_model.save_weights(conf.NNs_path+"/N_try_{}/actor_{}.h5".format(self.N_try,update_step_counter))
        self.critic_model.save_weights(conf.NNs_path+"/N_try_{}/critic_{}.h5".format(self.N_try,update_step_counter))
        self.target_critic.save_weights(conf.NNs_path+"/N_try_{}/target_critic_{}.h5".format(self.N_try,update_step_counter))

    def create_TO_init(self, ep, ICS):
        ''' Create initial state and initial controls for TO '''
        self.init_rand_state = ICS    
        
        self.NSTEPS_SH = conf.NSTEPS - int(self.init_rand_state[-1]/conf.dt)
        if self.NSTEPS_SH == 0:
            return None, None, None, None, 0

        # Initialize array to store RL state, control, and end-effector trajectories
        self.control_arr = np.empty((self.NSTEPS_SH, conf.nb_action))
        self.state_arr = np.empty((self.NSTEPS_SH+1, conf.nb_state))
        self.ee_pos_arr = np.empty((self.NSTEPS_SH+1,3))

        # Set initial state and end-effector position
        self.state_arr[0,:] = self.init_rand_state
        self.ee_pos_arr[0,:] = self.env.get_end_effector_position(self.state_arr[0, :])

        # Initialize array to initialize TO state and control variables
        init_TO_controls = np.zeros((self.NSTEPS_SH, conf.nb_action))
        init_TO_states = np.zeros(( self.NSTEPS_SH+1, conf.nb_state))

        # Set initial state 
        init_TO_states[0,:] = self.init_rand_state

        # Simulate actor's actions to compute the state trajectory used to initialize TO state variables (use ICS for state and 0 for control if it is the first episode otherwise use policy rollout)
        success_init_flag = 1
        for i in range(self.NSTEPS_SH):   
            if ep == 0:
                init_TO_controls[i,:] = np.zeros(conf.nb_action)
            else:
                init_TO_controls[i,:] = tf.squeeze(self.NN.eval(self.actor_model, np.array([init_TO_states[i,:]]))).numpy()
            init_TO_states[i+1,:] = self.env.simulate(init_TO_states[i,:],init_TO_controls[i,:])
            if np.isnan(init_TO_states[i+1,:]).any():
                success_init_flag = 0
                return None, None, None, None, success_init_flag

        return self.init_rand_state, init_TO_states, init_TO_controls, self.NSTEPS_SH, success_init_flag

class RL_AC_torch:
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
    
    def setup_model(self, recover_training=None, weights=None):
        ''' Setup RL model '''
        # Create actor, critic and target NNs
        if weights is not None:
            self.actor_model = self.NN.create_actor(weights=weights[0])
            self.critic_model = self.NN.create_critic_sine(weights=weights[1])
            self.target_critic = self.NN.create_critic_sine(weights=weights[2])
        else:
            self.actor_model = self.NN.create_actor()
            self.critic_model = self.NN.create_critic_sine()
            self.target_critic = self.NN.create_critic_sine()


        self.critic_optimizer   = torch.optim.Adam(self.critic_model.parameters(), eps = 1e-7, lr = conf.CRITIC_LEARNING_RATE)
        self.actor_optimizer    = torch.optim.Adam(self.actor_model.parameters(), eps = 1e-7, lr = conf.ACTOR_LEARNING_RATE)
        # Set optimizer specifying the learning rates
        if conf.LR_SCHEDULE:
            # Piecewise constant decay schedule

            #NOTE: not sure about epochs used in 'milestones' variable

            self.CRITIC_LR_SCHEDULE = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones = conf.values_schedule_LR_C, gamma = 0.5)
            self.ACTOR_LR_SCHEDULE  = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones = conf.values_schedule_LR_A, gamma = 0.5)

        # Set initial weights of the NNs
        if recover_training is not None: 
            #NOTE: this was not tested
            NNs_path_rec = str(recover_training[0])
            N_try = recover_training[1]
            update_step_counter = recover_training[2]   
            self.actor_model.load_state_dict(torch.load(f"{NNs_path_rec}/N_try_{N_try}/actor_{update_step_counter}.pt"))
            self.critic_model.load_state_dict(torch.load(f"{NNs_path_rec}/N_try_{N_try}/critic_{update_step_counter}.pt"))
            self.target_critic.load_state_dict(torch.load(f"{NNs_path_rec}/N_try_{N_try}/target_critic_{update_step_counter}.pt"))
        else:
            self.target_critic.load_state_dict(self.critic_model.state_dict())   

    def update(self, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, term_batch, weights_batch, batch_size=None):
        ''' Update both critic and actor '''
        # Ensure models are in training mode
        #self.actor_model.train()
        #self.critic_model.train()

        # Update the critic by backpropagating the gradients
        self.critic_optimizer.zero_grad()
        reward_to_go_batch, critic_value, target_critic_value = self.NN.compute_critic_grad(self.critic_model, self.target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch)
        #critic_grad.backward()  # Compute the gradients
        self.critic_optimizer.step()  # Update the weights
        
        # Update the actor by backpropagating the gradients
        self.actor_optimizer.zero_grad()
        self.NN.compute_actor_grad(self.actor_model, self.critic_model, state_batch, term_batch, batch_size)
        #actor_grad.backward()  # Compute the gradients
        self.actor_optimizer.step()  # Update the weights
        if conf.LR_SCHEDULE:
            self.ACTOR_LR_SCHEDULE.step()
            self.CRITIC_LR_SCHEDULE.step()

        return reward_to_go_batch, critic_value, target_critic_value
        
    #@tf.function
    def update_target(self, target_weights, weights):
        ''' Update target critic NN '''
        tau = conf.UPDATE_RATE
        with torch.no_grad():
            for target_param, param in zip(target_weights, weights):
                target_param.data.copy_(param.data * tau + target_param.data * (1 - tau))

    def learn_and_update(self, update_step_counter, buffer, ep):
        #Tested Successfully# Although only for one iteration (?)
        ''' Sample experience and update buffer priorities and NNs '''
        for _ in range(int(self.conf.UPDATE_LOOPS[ep])):
            # Sample batch of transitions from the buffer
            state_batch, partial_reward_to_go_batch, state_next_rollout_batch, dVdx_batch, d_batch, term_batch, weights_batch, batch_idxes = buffer.sample()

            # Update both critic and actor
            reward_to_go_batch, critic_value, target_critic_value = self.update(state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, term_batch, weights_batch)

            # Update buffer priorities
            if self.conf.prioritized_replay_alpha != 0:
                buffer.update_priorities(batch_idxes, reward_to_go_batch, critic_value, target_critic_value)

            # Update target critic
            if not self.conf.MC:
                self.update_target(self.target_critic.parameters(), self.critic_model.parameters())

            update_step_counter += 1

            # Plot rollouts and save the NNs every conf.save_interval training episodes
            if update_step_counter % self.conf.save_interval == 0:
                self.RL_save_weights(update_step_counter)

        return update_step_counter
    
    def RL_Solve(self, TO_controls, TO_states, TO_step_cost):
        ''' Solve RL problem '''
        ep_return = 0                                                                 # Initialize the return
        rwrd_arr = np.empty(self.NSTEPS_SH+1)                                         # Reward array
        state_next_rollout_arr = np.zeros((self.NSTEPS_SH+1, conf.nb_state))     # Next state array
        partial_reward_to_go_arr = np.empty(self.NSTEPS_SH+1)                         # Partial cost-to-go array
        total_reward_to_go_arr = np.empty(self.NSTEPS_SH+1)                           # Total cost-to-go array
        term_arr = np.zeros(self.NSTEPS_SH+1)                                         # Episode-termination flag array
        term_arr[-1] = 1
        done_arr = np.zeros(self.NSTEPS_SH+1)                                         # Episode-MC-termination flag array

        # START RL EPISODE
        self.control_arr = TO_controls # action clipped in TO
        
        if conf.env_RL:
            for step_counter in range(self.NSTEPS_SH):
                # Simulate actions and retrieve next state and compute reward
                self.state_arr[step_counter+1,:], rwrd_arr[step_counter] = self.env.step(conf.cost_weights_running, self.state_arr[step_counter,:], self.control_arr[step_counter,:])

                # Compute end-effector position
                self.ee_pos_arr[step_counter+1,:] = self.env.get_end_effector_position(self.state_arr[step_counter+1, :])
            rwrd_arr[-1] = self.env.reward(conf.cost_weights_terminal, self.state_arr[-1,:])
        else:
            self.state_arr, rwrd_arr = TO_states, -TO_step_cost

        ep_return = sum(rwrd_arr)

        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        for i in range(self.NSTEPS_SH+1):
            # set final lookahead step depending on whether Monte Cartlo or TD(n) is used
            if conf.MC:
                final_lookahead_step = self.NSTEPS_SH
                done_arr[i] = 1 
            else:
                final_lookahead_step = min(i+conf.nsteps_TD_N, self.NSTEPS_SH)
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
        actor_model_path = f"{conf.NNs_path}/N_try_{self.N_try}/actor_{update_step_counter}.pth"
        critic_model_path = f"{conf.NNs_path}/N_try_{self.N_try}/critic_{update_step_counter}.pth"
        target_critic_path = f"{conf.NNs_path}/N_try_{self.N_try}/target_critic_{update_step_counter}.pth"

        # Save model weights
        torch.save(self.actor_model.state_dict(), actor_model_path)
        torch.save(self.critic_model.state_dict(), critic_model_path)
        torch.save(self.target_critic.state_dict(), target_critic_path)

    def create_TO_init(self, ep, ICS):
        ''' Create initial state and initial controls for TO '''
        self.init_rand_state = ICS    
        
        self.NSTEPS_SH = conf.NSTEPS - int(self.init_rand_state[-1]/conf.dt)
        if self.NSTEPS_SH == 0:
            return None, None, None, None, 0

        # Initialize array to store RL state, control, and end-effector trajectories
        self.control_arr = np.empty((self.NSTEPS_SH, conf.nb_action))
        self.state_arr = np.empty((self.NSTEPS_SH+1, conf.nb_state))
        self.ee_pos_arr = np.empty((self.NSTEPS_SH+1,3))

        # Set initial state and end-effector position
        self.state_arr[0,:] = self.init_rand_state
        self.ee_pos_arr[0,:] = self.env.get_end_effector_position(self.state_arr[0, :])

        # Initialize array to initialize TO state and control variables
        init_TO_controls = np.zeros((self.NSTEPS_SH, conf.nb_action))
        init_TO_states = np.zeros(( self.NSTEPS_SH+1, conf.nb_state))

        # Set initial state 
        init_TO_states[0,:] = self.init_rand_state

        # Simulate actor's actions to compute the state trajectory used to initialize TO state variables (use ICS for state and 0 for control if it is the first episode otherwise use policy rollout)
        success_init_flag = 1
        for i in range(self.NSTEPS_SH):   
            if ep == 0:
                init_TO_controls[i,:] = np.zeros(conf.nb_action)
            else:
                #init_TO_controls[i,:] = tf.squeeze(self.NN.eval(self.actor_model, np.array([init_TO_states[i,:]]))).numpy()
                init_TO_controls[i,:] = self.NN.eval(self.actor_model, torch.tensor(np.array([init_TO_states[i,:]]), dtype=torch.float32)).squeeze().detach().numpy()
                #print(init_TO_controls.shape)
            init_TO_states[i+1,:] = self.env.simulate(init_TO_states[i,:],init_TO_controls[i,:])
            if np.isnan(init_TO_states[i+1,:]).any():
                success_init_flag = 0
                return None, None, None, None, success_init_flag

        return self.init_rand_state, init_TO_states, init_TO_controls, self.NSTEPS_SH, success_init_flag

class Envtorch:
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
        state = np.zeros(conf.nb_state)

        time = np.random.uniform(conf.x_init_min[-1], conf.x_init_max[-1])
        for i in range(conf.nb_state-1): 
            state[i] = np.random.uniform(conf.x_init_min[i], conf.x_init_max[i])
        state[-1] = conf.dt*round(time/conf.dt)
        return state

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

        return torch.tensor(state_next, dtype=torch.float32)
        
    def derivative_batch(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        Fu = np.array([self.derivative(s, a) for s, a in zip(state, action)])

        return torch.tensor(Fu, dtype=torch.float32)
    
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

class DoubleIntegratortorch(Envtorch):
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
        partial_reward = torch.tensor([self.reward(w, s) for w, s in zip(weights, state)], dtype=torch.float32)

        # Redefine action-related cost
        
        act_sq = torch.pow(action,2)
        norm_act_e10 = torch.pow(action/torch.tensor(self.conf.u_max), 10)
        u_cost = torch.sum((act_sq + self.conf.w_b*norm_act_e10), dim=1)
        weights = torch.tensor(weights, dtype=torch.float32)
        
        r = self.scale*(-weights[:,6]*u_cost) + partial_reward
        return torch.reshape(r, (r.shape[0], 1))

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
        self.offset = conf.cost_funct_param[0]
        self.scale = conf.cost_funct_param[1]

    def reset(self):
        ''' Choose initial state uniformly at random '''
        state = np.zeros(conf.nb_state)

        time = np.random.uniform(conf.x_init_min[-1], conf.x_init_max[-1])
        for i in range(conf.nb_state-1): 
            state[i] = np.random.uniform(conf.x_init_min[i], conf.x_init_max[i])
        state[-1] = conf.dt*round(time/conf.dt)
        return state

    def check_ICS_feasible(self, state):
        ''' Check if ICS is not feasible '''
        # check if ee is in the obstacles
        p_ee = self.get_end_effector_position(state)

        ellipse1 = ((p_ee[0] - conf.XC1)**2) / ((conf.A1 / 2)**2) + ((p_ee[1] - conf.YC1)**2) / ((conf.B1 / 2)**2)
        ellipse2 = ((p_ee[0] - conf.XC2)**2) / ((conf.A2 / 2)**2) + ((p_ee[1] - conf.YC2)**2) / ((conf.B2 / 2)**2)
        ellipse3 = ((p_ee[0] - conf.XC3)**2) / ((conf.A3 / 2)**2) + ((p_ee[1] - conf.YC3)**2) / ((conf.B3 / 2)**2)
        
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
        conf.simu.simulate(np.copy(state[:-1]), action, conf.dt, 1) ### Explicit Euler ###

        # Return next state
        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(conf.simu.q), np.copy(conf.simu.v)
        state_next[-1] = state[-1] + conf.dt
        
        return state_next
    
    def derivative(self, state, action):
        ''' Compute the derivative '''
        # Create robot model in Pinocchio with q_init as initial configuration
        q_init = state[:self.nq]
        v_init = state[self.nq:self.nx]

        # Dynamics gradient w.r.t control (1st order euler)
        pin.computeABADerivatives(conf.robot.model, conf.robot.data, np.copy(q_init), np.copy(v_init), action)       

        Fu = np.zeros((self.nx+1, self.nu))
        Fu[self.nv:-1, :] = conf.robot.data.Minv
        Fu[:self.nx, :] *= conf.dt

        if conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1/conf.state_norm_arr[:-1,None])  

        return Fu
    
    def augmented_derivative(self, state, action):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        q = state[:self.nq]
        v = state[self.nq:self.nx]
                
        # Compute Jacobians for continuous time dynamics
        Fx = np.zeros((conf.nb_state-1,conf.nb_state-1))
        Fu = np.zeros((conf.nb_state-1,conf.nb_action))

        pin.computeABADerivatives(conf.robot.model, conf.robot.data, q, v, action)

        Fx[:self.nv, :self.nv] = 0.0
        Fx[:self.nv, self.nv:self.nx] = np.identity(self.nv)
        Fx[self.nv:self.nx, :self.nv] = conf.robot.data.ddq_dq
        Fx[self.nv:self.nx, self.nv:self.nx] = conf.robot.data.ddq_dv
        Fu[self.nv:self.nx, :] = conf.robot.data.Minv
        
        # Convert them to discrete time
        Fx = np.identity(conf.nb_state-1) + conf.dt * Fx
        Fu *= conf.dt
        
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

        RF = conf.robot.model.getFrameId(conf.end_effector_frame_id) 

        H = conf.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p
    
    def bound_control_cost(self, action):
        u_cost = 0
        for i in range(conf.nb_action):
            u_cost += action[i]*action[i] + conf.w_b*(action[i]/conf.u_max[i])**10
        
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

        conf = conf

        super().__init__(conf)

        # Rename reward parameters
        self.offset = conf.cost_funct_param[0]
        self.scale = conf.cost_funct_param[1]

        self.alpha = conf.soft_max_param[0]
        self.alpha2 = conf.soft_max_param[1]

        self.XC1 = conf.obs_param[0]
        self.YC1 = conf.obs_param[1]
        self.XC2 = conf.obs_param[2]
        self.YC2 = conf.obs_param[3]
        self.XC3 = conf.obs_param[4]
        self.YC3 = conf.obs_param[5]
        
        self.A1 = conf.obs_param[6]
        self.B1 = conf.obs_param[7]
        self.A2 = conf.obs_param[8]
        self.B2 = conf.obs_param[9]
        self.A3 = conf.obs_param[10]
        self.B3 = conf.obs_param[11]

        self.TARGET_STATE = conf.TARGET_STATE
    
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
        u_cost = tf.reduce_sum((action**2 + conf.w_b*(action/conf.u_max)**10),axis=1) 

        r = self.scale*(- weights[:,6]*u_cost) + tf.convert_to_tensor(partial_reward, dtype=tf.float32)

        return tf.reshape(r, [r.shape[0], 1])

class PLOT_tf():
    def __init__(self, N_try, env, NN, conf):
        '''    
        :input N_try :                          (Test number)

        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)
            :param fig_ax_lim :                 (float array) Figure axis limit [x_min, x_max, y_min, y_max]
            :param Fig_path :                   (str) Figure path
            :param NSTEPS :                     (int) Max episode length
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param NORMALIZE_INPUTS :           (bool) Flag to normalize inputs (state)
            :param state_norm_array :           (float array) Array used to normalize states
            :param dt :                         (float) Timestep
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
        '''
        self.env = env  
        self.NN = NN     
        self.conf = conf

        self.N_try = N_try

        self.xlim = conf.fig_ax_lim[0].tolist()
        self.ylim = conf.fig_ax_lim[1].tolist()

        # Set the ticklabel font size globally
        plt.rcParams['xtick.labelsize'] = 22
        plt.rcParams['ytick.labelsize'] = 22
        plt.rcParams.update({'font.size': 20})

        return 

    def plot_obstaces(self, a=1):
        if self.conf.system_id == 'car_park':
            obs1 = Rectangle((self.conf.XC1-self.conf.A1/2, self.conf.YC1-self.conf.B1/2), self.conf.A1, self.conf.B1, 0.0,alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Rectangle((self.conf.XC2-self.conf.A2/2, self.conf.YC2-self.conf.B2/2), self.conf.A2, self.conf.B2, 0.0,alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Rectangle((self.conf.XC3-self.conf.A3/2, self.conf.YC3-self.conf.B3/2), self.conf.A3, self.conf.B3, 0.0,alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])

            #rec1 = FancyBboxPatch((self.conf.XC1-self.conf.A1/2, self.conf.YC1-self.conf.B1/2), self.conf.A1, self.conf.B1,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec1.set_facecolor([30/255, 130/255, 76/255, 1])
            #rec2 = FancyBboxPatch((self.conf.XC2-self.conf.A2/2, self.conf.YC2-self.conf.B2/2), self.conf.A2, self.conf.B2,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec2.set_facecolor([30/255, 130/255, 76/255, 1])
            #rec3 = FancyBboxPatch((self.conf.XC3-self.conf.A3/2, self.conf.YC3-self.conf.B3/2), self.conf.A3, self.conf.B3,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec3.set_facecolor([30/255, 130/255, 76/255, 1])
        else:
            obs1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, 0.0,alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, 0.0,alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, 0.0,alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])

        return [obs1, obs2, obs3]

    def plot_Reward(self, plot_obs=0):
        x = np.arange(-15, 15, 0.1)
        y = np.arange(-10, 10, 0.1)
        theta = np.pi/2
        ICS = np.array([np.array([i,j,0]) for i in x for j in y])
        state = np.array([self.compute_ICS(np.array([i,j,0]), 'car')[0] for i in x for j in y]) # for k in theta]
        state[:,2] = theta
        r = [self.env.reward(self.conf.cost_weights_running, s) for s in state]
        mi = min(r)
        ma = max(r)
        norm = colors.Normalize(vmin=mi,vmax=ma)
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        pti = ax.scatter(ICS[:,0], ICS[:,1], norm=norm, c=r, cmap=cm.get_cmap('hot_r'))
        plt.colorbar(pti)

        if plot_obs:
            obs_plot_list = self.plot_obstaces()
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
        
        # Center and check points of 'car_park' system
        #check_points_WF_i = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).dot(self.conf.check_points_BF[0,:]) + ICS[0,:2]
        #ax.scatter(check_points_WF_i[0], check_points_WF_i[1], c='b')
        #for i in range(1,len(self.conf.check_points_BF)):
        #    check_points_WF_i = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).dot(self.conf.check_points_BF[i,:]) + ICS[0,:2]
        #    ax.scatter(check_points_WF_i[0], check_points_WF_i[1], c='r')

        ax.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5, legend='Goal position') 
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Plane')
        #ax.legend()
        ax.grid(True)
        plt.show()
    
    def compute_ICS(self, p_ee, sys_id, theta=None, continue_flag=0):
        if sys_id == 'manipulator':
            radius = math.sqrt((p_ee[0]-self.conf.x_base)**2+(p_ee[1])**2)
            if radius > 30:
                continue_flag = 1
                return None, continue_flag

            phi = math.atan2(p_ee[1]-self.conf.y_base,(p_ee[0]-self.conf.x_base))               # SUM OF THE ANGLES FIXED   
            X3rd_joint = (p_ee[0]-self.conf.x_base) - self.conf.l* math.cos(phi) 
            Y3rd_joint = (p_ee[1]-self.conf.y_base) - self.conf.l* math.sin(phi)

            if abs(X3rd_joint) <= 1e-6 and abs(Y3rd_joint) <= 1e-6:
                continue_flag = 1
                return None, continue_flag

            c2 = (X3rd_joint**2 + Y3rd_joint**2 -2*self.conf.l**2)/(2*self.conf.l**2)

            if p_ee[1] >= 0:
                s2 = math.sqrt(1-c2**2)
            else:
                s2 = -math.sqrt(1-c2**2)

            s1 = ((self.conf.l + self.conf.l*c2)*Y3rd_joint - self.conf.l*s2*X3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)  
            c1 = ((self.conf.l + self.conf.l*c2)*X3rd_joint - self.conf.l*s2*Y3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)
            ICS_q0 = math.atan2(s1,c1)
            ICS_q1 = math.atan2(s2,c2)
            ICS_q2 = phi-ICS_q0-ICS_q1

            ICS = np.array([ICS_q0, ICS_q1, ICS_q2, 0.0, 0.0, 0.0, 0.0])

        elif sys_id == 'car':
            if theta == None:
                theta = 0*np.random.uniform(-math.pi,math.pi)
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'car_park':
            if theta == None:
                #theta = 0*np.random.uniform(-math.pi,math.pi)
                theta = np.pi/2
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'double_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0, 0.0, 0.0])
        
        elif sys_id == 'single_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0])
        
        return ICS, continue_flag

    def plot_policy(self, tau, x, y, steps, n_updates, diff_loc=0):
        ''' Plot policy rollout from a single initial state as well as state and control trajectories '''
        timesteps = self.self.conf.dt*np.arange(steps)
        
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.N_try,n_updates), y=1)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, x, 'ro', linewidth=1, markersize=1, legedn='x') 
        ax1.plot(timesteps, y, 'bo', linewidth=1, markersize=1, legend='y')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('[m]')  
        ax1.set_title('End-Effector Position') 
        ax1.legend()
        ax1.grid(True) 

        col = ['ro', 'bo', 'go']
        ax2 = fig.add_subplot(2, 2, self.conf.nb_action)
        for i in range(self.conf.nb_action):
            ax2.plot(timesteps, tau[:,i], col[i], linewidth=1, markersize=1,legend='tau{}'.format(i)) 
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Controls')
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(x, y, 'ro', linewidth=1, markersize=1) 
        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax3.add_artist(obs_plot_list[i]) 
        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=10) 
        ax3.set_xlim(self.xlim)
        ax3.set_ylim(self.ylim)
        ax3.set_aspect('equal', 'box')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_title('Plane')
        ax3.grid(True)

        fig.tight_layout()

        if diff_loc==0:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationSingleInit_{}_{}'.format(self.N_try,n_updates))
        else:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_{}_{}'.format(self.N_try,n_updates))

        plt.clf()
        plt.close(fig)

    def plot_policy_eval(self, p_list, n_updates, diff_loc=0, theta=0):
        ''' Plot only policy rollouts from multiple initial states '''
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.N_try,n_updates), y=1)

        ax = fig.add_subplot(1, 1, 1)
        for idx in range(len(p_list)):
            ax.plot(p_list[idx][:,0], p_list[idx][:,1], marker='o', linewidth=0.3, markersize=1)
            ax.plot(p_list[idx][0,0],p_list[idx][0,1],'ko',markersize=5)
            if self.conf.system_id == 'car_park':
                theta = p_list[idx][-1,2]
                fancybox = FancyBboxPatch((0 - self.conf.L/2, 0 - self.conf.W/2), self.conf.L, self.conf.W, edgecolor='none', alpha=0.5, boxstyle='round,pad=0')
                fancybox.set_transform(Affine2D().rotate_deg(np.rad2deg(theta)).translate(p_list[idx][-1,0], p_list[idx][-1,1]) + ax.transData)
                ax.add_patch(fancybox)

        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax.add_artist(obs_plot_list[i]) 

        ax.plot(self.conf.TARGET_STATE[0],self.conf.TARGET_STATE[1],'b*',markersize=10)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True)
        fig.tight_layout()
        if diff_loc==0:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationSingleInit_{}_{}_TF'.format(self.N_try,n_updates))
        else:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_{}_{}_TF'.format(self.N_try,n_updates))

        plt.clf()
        plt.close(fig)

    def rollout(self,update_step_cntr, actor_model, init_states_sim, diff_loc=0):
        ''' Plot rollout of the actor from some initial states. It generates the results and then calls plot_policy() and plot_policy_eval() '''
        #tau_all_sim = []
        p_ee_all_sim = []

        returns = {}

        for k in range(len(init_states_sim)):
            rollout_controls = np.zeros((self.conf.NSTEPS,self.conf.nb_action))
            rollout_states = np.zeros((self.conf.NSTEPS+1,self.conf.nb_state))
            rollout_p_ee = np.zeros((self.conf.NSTEPS+1,3))
            rollout_episodic_reward = 0

            rollout_p_ee[0,:] = self.env.get_end_effector_position(init_states_sim[k])
            rollout_states[0,:] = np.copy(init_states_sim[k])
            
            for i in range(self.conf.NSTEPS):
                rollout_controls[i,:] = tf.squeeze(self.NN.eval(actor_model, np.array([rollout_states[i,:]]))).numpy()
                rollout_states[i+1,:], rwrd_sim = self.env.step(self.conf.cost_weights_running, rollout_states[i,:],rollout_controls[i,:])
                rollout_p_ee[i+1,:] = self.env.get_end_effector_position(rollout_states[i+1,:])
                
                rollout_p_ee[i+1,-1] = rollout_states[i+1,2] ### !!! ###

                rollout_episodic_reward += rwrd_sim

            if k==0:
                print("N try = {}: Simulation Return @ N updates = {} ==> {}".format(self.N_try,update_step_cntr,rollout_episodic_reward))
                
            p_ee_all_sim.append(rollout_p_ee)  

            returns[init_states_sim[k][0],init_states_sim[k][1]] = rollout_episodic_reward

        self.plot_policy_eval(p_ee_all_sim,update_step_cntr, diff_loc=diff_loc)

        return returns

    def plot_results(self, tau, ee_pos_TO, ee_pos_RL, steps, to=0):
        ''' Plot results from TO and episode to check consistency '''
        timesteps = self.conf.dt*np.arange(steps+1)
        fig = plt.figure(figsize=(12,8))
        if to:
            plt.suptitle('TO EXPLORATION: N try = {}'.format(self.N_try), y=1, fontsize=20)
        else:  
            plt.suptitle('POLICY EXPLORATION: N try = {}'.format(self.N_try), y=1, fontsize=20)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, ee_pos_TO[:,0], 'ro', linewidth=1, markersize=1,legend="x_TO") 
        ax1.plot(timesteps, ee_pos_TO[:,1], 'bo', linewidth=1, markersize=1,legend="y_TO")
        ax1.plot(timesteps, ee_pos_RL[:,0], 'go', linewidth=1, markersize=1,legend="x_RL") 
        ax1.plot(timesteps, ee_pos_RL[:,1], 'ko', linewidth=1, markersize=1,legend="y_RL")
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('[m]')    
        ax1.set_title('End-Effector Position')
        ax1.set_xlim(0, timesteps[-1])
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 2, 3)
        col = ['ro', 'bo', 'go']
        for i in range(self.conf.nb_action):
            ax2.plot(timesteps[:-1], tau[:,i], col[i], linewidth=1, markersize=1,legend='tau{}'.format(i)) 
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Controls')
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(ee_pos_TO[:,0], ee_pos_TO[:,1], 'ro', linewidth=1, markersize=2,legend='TO')
        ax3.plot(ee_pos_RL[:,0], ee_pos_RL[:,1], 'bo', linewidth=1, markersize=2,legend='RL')
        ax3.plot([ee_pos_TO[0,0]],[ee_pos_TO[0,1]],'ro',markersize=5)
        ax3.plot([ee_pos_RL[0,0]],[ee_pos_RL[0,1]],'bo',markersize=5)
        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax3.add_artist(obs_plot_list[i]) 
        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
        ax3.set_xlim(self.xlim)
        ax3.set_ylim(self.ylim)
        ax3.set_aspect('equal', 'box')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_title('Plane')
        ax3.legend()
        ax3.grid(True)

        fig.tight_layout()
        #plt.show()

    def plot_Return(self, ep_reward_list):
        ''' Plot returns (not so meaningful given that the initial state, so also the time horizon, of each episode is randomized) '''
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)   
        ax.set_yscale('log') 
        ax.plot(ep_reward_list**2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title("N_try = {}".format(self.N_try))
        ax.grid(True)
        plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/EpReturn_{}_TF'.format(self.N_try))
        plt.close()

    def plot_Critic_Value_function(self, critic_model, n_update, sys_id, name='V_TF_'):
        ''' Plot Value function as learned by the critic '''
        if sys_id == 'manipulator':
            N_discretization_x = 60 + 1  
            N_discretization_y = 60 + 1

            plot_data = np.zeros(N_discretization_y*N_discretization_x)*np.nan
            ee_pos = np.zeros((N_discretization_y*N_discretization_x,3))*np.nan

            for k_x in range(N_discretization_x):
                for k_y in range(N_discretization_y):
                    ICS = self.env.reset()
                    ICS[-1] = 0
                    ee_pos[k_x*(N_discretization_y)+k_y,:] = self.env.get_end_effector_position(ICS)
                    plot_data[k_x*(N_discretization_y)+k_y] = self.NN.eval(critic_model, np.array([ICS]))

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot()
            plt.scatter(ee_pos[:,0], ee_pos[:,1], c=plot_data, cmap=cm.coolwarm, antialiased=False)
            obs_plot_list = self.plot_obstaces(a=0.5)
            for i in range(len(obs_plot_list)):
                ax.add_patch(obs_plot_list[i])
            plt.colorbar()
            plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', 'box')
            plt.savefig('{}/N_try_{}/{}_{}'.format(self.conf.Fig_path,self.N_try,name,int(n_update)))
            plt.close()

        else:
            N_discretization_x = 30 + 1  
            N_discretization_y = 30 + 1

            plot_data = np.zeros((N_discretization_y,N_discretization_x))*np.nan

            ee_x = np.linspace(-15, 15, N_discretization_x)
            ee_y = np.linspace(-15, 15, N_discretization_y)

            for k_y in range(N_discretization_y):
                for k_x in range(N_discretization_x):
                    p_ee = np.array([ee_x[k_x], ee_y[k_y], 0])
                    ICS, continue_flag = self.compute_ICS(p_ee, sys_id, continue_flag=0)
                    if continue_flag:
                        continue
                    plot_data[k_x,k_y] = self.NN.eval(critic_model, np.array([ICS]))

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot()
            plt.contourf(ee_x, ee_y, plot_data.T, cmap=cm.coolwarm, antialiased=False)

            obs_plot_list = self.plot_obstaces(a=0.5)
            for i in range(len(obs_plot_list)):
                ax.add_patch(obs_plot_list[i])
            plt.colorbar()
            plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', 'box')
            plt.savefig('{}/N_try_{}/{}_{}'.format(self.conf.Fig_path,self.N_try,name,int(n_update)))
            plt.close()

    def plot_Critic_Value_function_from_sample(self, n_update, NSTEPS_SH, state_arr, reward_arr):
        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        reward_to_go_arr = np.zeros(sum(NSTEPS_SH)+len(NSTEPS_SH)*1)
        idx = 0
        for n in range(len(NSTEPS_SH)):
            for i in range(NSTEPS_SH[n]+1):
                # Compute the partial cost to go
                reward_to_go_arr[idx] = sum(reward_arr[n][i:])
                idx += 1

        state_arr = np.concatenate(state_arr, axis=0)
        ee_pos_arr = np.zeros((len(state_arr),3))
        for i in range(state_arr.shape[0]):
            ee_pos_arr[i,:] = self.env.get_end_effector_position(state_arr[i])
        

        mi = min(reward_to_go_arr)
        ma = max(reward_to_go_arr)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()#projection='3d')
        norm = colors.Normalize(vmin=mi,vmax=ma)

        obs_plot_list = self.plot_obstaces(a=0.5)
        
        ax.scatter(ee_pos_arr[:,0],ee_pos_arr[:,1], c=reward_to_go_arr, norm=norm, cmap=cm.coolwarm, marker='x')
        
        for i in range(len(obs_plot_list)):
            ax.add_patch(obs_plot_list[i])

        plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.coolwarm))
        plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        plt.savefig('{}/N_try_{}/V_sample_{}'.format(self.conf.Fig_path,self.N_try,int(n_update)))
        plt.close()

    def plot_ICS(self,state_arr):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        for j in range(len(state_arr)):
            ax.scatter(state_arr[j][0,0],state_arr[j][0,1])
            obs_plot_list = plot_fun.plot_obstaces()
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
        ax.set_xlim(self.fig_ax_lim[0].tolist())
        ax.set_ylim(self.fig_ax_lim[1].tolist())
        ax.set_aspect('equal', 'box')
        plt.savefig('{}/N_try_{}/ICS_{}_S{}'.format(conf.Fig_path,N_try,update_step_counter,int(w_S)))
        plt.close(fig)

    def plot_rollout_and_traj_from_ICS(self, init_state, n_update, actor_model, TrOp, tag, steps=200):
        ''' Plot results from TO and episode to check consistency '''
        colors = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        
        for j in range(len(init_state)):

            ee_pos_TO = np.zeros((steps,3))
            ee_pos_RL = np.zeros((steps,3))

            RL_states = np.zeros((steps,self.conf.nb_state))
            RL_action = np.zeros((steps-1,self.conf.nb_action))
            RL_states[0,:] = init_state[j,:]
            ee_pos_RL[0,:] = self.env.get_end_effector_position(RL_states[0,:])

            for i in range(steps-1):
                RL_action[i,:] = self.NN.eval(actor_model, torch.tensor(np.array([RL_states[i,:]]), dtype=torch.float32))
                RL_states[i+1,:] = self.env.simulate(RL_states[i,:], RL_action[i,:])
                ee_pos_RL[i+1,:] = self.env.get_end_effector_position(RL_states[i+1,:])
            
            TO_states, _ = TrOp.TO_System_Solve3(init_state[j,:], RL_states.T, RL_action.T, steps-1)

            try:
                for i in range(steps):
                    ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[i,:])
            except:
                ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[0,:])
                
            ax.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax.scatter(ee_pos_TO[0,0],ee_pos_TO[0,1],color=colors[j])
            ax.scatter(ee_pos_RL[0,0],ee_pos_RL[0,1],color=colors[j])
            ax.plot(ee_pos_TO[1:,0],ee_pos_TO[1:,1],color=colors[j])
            ax.plot(ee_pos_RL[1:,0],ee_pos_RL[1:,1],'--',color=colors[j])
        
        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax.add_patch(obs_plot_list[i])

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Plane')
        #ax.legend()
        ax.grid(True)

        plt.savefig('{}/N_try_{}/ee_traj_{}_{}'.format(self.conf.Fig_path,self.N_try,int(n_update), tag))

    def plot_ICS(self, input_arr, cs=0):
        if cs == 1:
            p_arr = np.zeros((len(input_arr),3))
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()
            for j in range(len(input_arr)):
                p_arr[j,:] = input_arr[j,:]
            ax.scatter(p_arr[:,0],p_arr[:,1])
            obs_plot_list = self.plot_obstaces(a = 0.5)
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
            ax.set_xlim(self.conf.fig_ax_lim[0].tolist())
            ax.set_ylim(self.conf.fig_ax_lim[1].tolist())
            ax.set_aspect('equal', 'box')
            ax.grid()
            plt.savefig('{}/N_try_{}/ICS'.format(self.conf.Fig_path,self.N_try))
            plt.close(fig)
        else:    
            p_arr = np.zeros((len(input_arr),3))
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()

            for j in range(len(input_arr)):
                p_arr[j,:] = self.env.get_end_effector_position(input_arr[j])
            ax.scatter(p_arr[:,0],p_arr[:,1])
            obs_plot_list = self.plot_obstaces(a = 0.5)
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
            ax.set_xlim(self.conf.fig_ax_lim[0].tolist())
            ax.set_ylim(self.conf.fig_ax_lim[1].tolist())
            ax.set_aspect('equal', 'box')
            ax.grid()
            plt.savefig('{}/N_try_{}/ICS'.format(self.conf.Fig_path,self.N_try))
            plt.close(fig)

    def plot_traj_from_ICS(self, init_state, TrOp, RLAC, update_step_counter=0,ep=0,steps=200, init=0,continue_flag=1):
        ''' Plot results from TO and episode to check consistency '''
        colors = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        for j in range(len(init_state)):

            ee_pos_TO = np.zeros((steps,3))
            ee_pos_RL = np.zeros((steps,3))
            
            if init == 0:
                # zeros
                _, init_TO_states, init_TO_controls, _, success_init_flag = RLAC.create_TO_init(0, init_state[j,:])
            elif init == 1:
                # NN
                _, init_TO_states, init_TO_controls, _, success_init_flag = RLAC.create_TO_init(1, init_state[j,:])

            if success_init_flag:
                _, _, TO_states, _, _, _  = TrOp.TO_System_Solve(init_state[j,:], init_TO_states, init_TO_controls, steps-1)
            else:
                continue

            try:
                for i in range(steps):
                    ee_pos_RL[i,:] = self.env.get_end_effector_position(init_TO_states[i,:])
                    ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[i,:])
            except:
                ee_pos_RL[i,:] = self.env.get_end_effector_position(init_TO_states[0,:])
                ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[0,:])

            ax1.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax1.scatter(ee_pos_RL[0,0],ee_pos_RL[0,1],color=colors[j])
            ax1.plot(ee_pos_RL[1:,0],ee_pos_RL[1:,1],'--',color=colors[j])
                
            ax2.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax2.scatter(ee_pos_TO[0,0],ee_pos_TO[0,1],color=colors[j])
            ax2.plot(ee_pos_TO[1:,0],ee_pos_TO[1:,1],color=colors[j])
        
        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax1.add_patch(obs_plot_list[i])

        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax2.add_patch(obs_plot_list[i])

        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)
        ax1.set_aspect('equal', 'box')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('Warmstart traj.')

        ax2.set_xlim(self.xlim)
        ax2.set_ylim(self.ylim)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X [m]')
        #ax2.set_ylabel('Y [m]')
        ax2.set_title('TO traj.')
        #ax.legend()
        ax1.grid(True)
        ax2.grid(True)

        plt.savefig('{}/N_try_{}/ee_traj_{}_{}_TF'.format(self.conf.Fig_path,self.N_try,init,update_step_counter))

class PLOT_torch():
    def __init__(self, N_try, env, NN, conf):
        '''    
        :input N_try :                          (Test number)

        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)
            :param fig_ax_lim :                 (float array) Figure axis limit [x_min, x_max, y_min, y_max]
            :param Fig_path :                   (str) Figure path
            :param NSTEPS :                     (int) Max episode length
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param NORMALIZE_INPUTS :           (bool) Flag to normalize inputs (state)
            :param state_norm_array :           (float array) Array used to normalize states
            :param dt :                         (float) Timestep
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
        '''
        self.env = env  
        self.NN = NN     
        self.conf = conf

        self.N_try = N_try

        self.xlim = conf.fig_ax_lim[0].tolist()
        self.ylim = conf.fig_ax_lim[1].tolist()

        # Set the ticklabel font size globally
        plt.rcParams['xtick.labelsize'] = 22
        plt.rcParams['ytick.labelsize'] = 22
        plt.rcParams.update({'font.size': 20})

        return 

    def plot_obstaces(self, a=1):
        if self.conf.system_id == 'car_park':
            obs1 = Rectangle((self.conf.XC1-self.conf.A1/2, self.conf.YC1-self.conf.B1/2), self.conf.A1, self.conf.B1, 0.0,alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Rectangle((self.conf.XC2-self.conf.A2/2, self.conf.YC2-self.conf.B2/2), self.conf.A2, self.conf.B2, 0.0,alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Rectangle((self.conf.XC3-self.conf.A3/2, self.conf.YC3-self.conf.B3/2), self.conf.A3, self.conf.B3, 0.0,alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])

            #rec1 = FancyBboxPatch((self.conf.XC1-self.conf.A1/2, self.conf.YC1-self.conf.B1/2), self.conf.A1, self.conf.B1,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec1.set_facecolor([30/255, 130/255, 76/255, 1])
            #rec2 = FancyBboxPatch((self.conf.XC2-self.conf.A2/2, self.conf.YC2-self.conf.B2/2), self.conf.A2, self.conf.B2,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec2.set_facecolor([30/255, 130/255, 76/255, 1])
            #rec3 = FancyBboxPatch((self.conf.XC3-self.conf.A3/2, self.conf.YC3-self.conf.B3/2), self.conf.A3, self.conf.B3,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec3.set_facecolor([30/255, 130/255, 76/255, 1])
        else:
            obs1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, 0.0,alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, 0.0,alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, 0.0,alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])

        return [obs1, obs2, obs3]

    def plot_Reward(self, plot_obs=0):
        x = np.arange(-15, 15, 0.1)
        y = np.arange(-10, 10, 0.1)
        theta = np.pi/2
        ICS = np.array([np.array([i,j,0]) for i in x for j in y])
        state = np.array([self.compute_ICS(np.array([i,j,0]), 'car')[0] for i in x for j in y]) # for k in theta]
        state[:,2] = theta
        r = [self.env.reward(self.conf.cost_weights_running, s) for s in state]
        mi = min(r)
        ma = max(r)
        norm = colors.Normalize(vmin=mi,vmax=ma)
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        pti = ax.scatter(ICS[:,0], ICS[:,1], norm=norm, c=r, cmap=cm.get_cmap('hot_r'))
        plt.colorbar(pti)

        if plot_obs:
            obs_plot_list = self.plot_obstaces()
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
        
        # Center and check points of 'car_park' system
        #check_points_WF_i = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).dot(self.conf.check_points_BF[0,:]) + ICS[0,:2]
        #ax.scatter(check_points_WF_i[0], check_points_WF_i[1], c='b')
        #for i in range(1,len(self.conf.check_points_BF)):
        #    check_points_WF_i = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).dot(self.conf.check_points_BF[i,:]) + ICS[0,:2]
        #    ax.scatter(check_points_WF_i[0], check_points_WF_i[1], c='r')

        ax.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5, legend='Goal position') 
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Plane')
        #ax.legend()
        ax.grid(True)
        plt.show()
    
    def compute_ICS(self, p_ee, sys_id, theta=None, continue_flag=0):
        if sys_id == 'manipulator':
            radius = math.sqrt((p_ee[0]-self.conf.x_base)**2+(p_ee[1])**2)
            if radius > 30:
                continue_flag = 1
                return None, continue_flag

            phi = math.atan2(p_ee[1]-self.conf.y_base,(p_ee[0]-self.conf.x_base))               # SUM OF THE ANGLES FIXED   
            X3rd_joint = (p_ee[0]-self.conf.x_base) - self.conf.l* math.cos(phi) 
            Y3rd_joint = (p_ee[1]-self.conf.y_base) - self.conf.l* math.sin(phi)

            if abs(X3rd_joint) <= 1e-6 and abs(Y3rd_joint) <= 1e-6:
                continue_flag = 1
                return None, continue_flag

            c2 = (X3rd_joint**2 + Y3rd_joint**2 -2*self.conf.l**2)/(2*self.conf.l**2)

            if p_ee[1] >= 0:
                s2 = math.sqrt(1-c2**2)
            else:
                s2 = -math.sqrt(1-c2**2)

            s1 = ((self.conf.l + self.conf.l*c2)*Y3rd_joint - self.conf.l*s2*X3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)  
            c1 = ((self.conf.l + self.conf.l*c2)*X3rd_joint - self.conf.l*s2*Y3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)
            ICS_q0 = math.atan2(s1,c1)
            ICS_q1 = math.atan2(s2,c2)
            ICS_q2 = phi-ICS_q0-ICS_q1

            ICS = np.array([ICS_q0, ICS_q1, ICS_q2, 0.0, 0.0, 0.0, 0.0])

        elif sys_id == 'car':
            if theta == None:
                theta = 0*np.random.uniform(-math.pi,math.pi)
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'car_park':
            if theta == None:
                #theta = 0*np.random.uniform(-math.pi,math.pi)
                theta = np.pi/2
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'double_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0, 0.0, 0.0])
        
        elif sys_id == 'single_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0])
        
        return ICS, continue_flag

    def plot_policy(self, tau, x, y, steps, n_updates, diff_loc=0):
        ''' Plot policy rollout from a single initial state as well as state and control trajectories '''
        timesteps = self.self.conf.dt*np.arange(steps)
        
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.N_try,n_updates), y=1)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, x, 'ro', linewidth=1, markersize=1, legedn='x') 
        ax1.plot(timesteps, y, 'bo', linewidth=1, markersize=1, legend='y')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('[m]')  
        ax1.set_title('End-Effector Position') 
        ax1.legend()
        ax1.grid(True) 

        col = ['ro', 'bo', 'go']
        ax2 = fig.add_subplot(2, 2, self.conf.nb_action)
        for i in range(self.conf.nb_action):
            ax2.plot(timesteps, tau[:,i], col[i], linewidth=1, markersize=1,legend='tau{}'.format(i)) 
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Controls')
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(x, y, 'ro', linewidth=1, markersize=1) 
        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax3.add_artist(obs_plot_list[i]) 
        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=10) 
        ax3.set_xlim(self.xlim)
        ax3.set_ylim(self.ylim)
        ax3.set_aspect('equal', 'box')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_title('Plane')
        ax3.grid(True)

        fig.tight_layout()

        if diff_loc==0:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationSingleInit_{}_{}'.format(self.N_try,n_updates))
        else:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_{}_{}'.format(self.N_try,n_updates))

        plt.clf()
        plt.close(fig)

    def plot_policy_eval(self, p_list, n_updates, diff_loc=0, theta=0):
        ''' Plot only policy rollouts from multiple initial states '''
        fig = plt.figure(figsize=(12,8))
        plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(self.N_try,n_updates), y=1)

        ax = fig.add_subplot(1, 1, 1)
        for idx in range(len(p_list)):
            ax.plot(p_list[idx][:,0], p_list[idx][:,1], marker='o', linewidth=0.3, markersize=1)
            ax.plot(p_list[idx][0,0],p_list[idx][0,1],'ko',markersize=5)
            if self.conf.system_id == 'car_park':
                theta = p_list[idx][-1,2]
                fancybox = FancyBboxPatch((0 - self.conf.L/2, 0 - self.conf.W/2), self.conf.L, self.conf.W, edgecolor='none', alpha=0.5, boxstyle='round,pad=0')
                fancybox.set_transform(Affine2D().rotate_deg(np.rad2deg(theta)).translate(p_list[idx][-1,0], p_list[idx][-1,1]) + ax.transData)
                ax.add_patch(fancybox)

        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax.add_artist(obs_plot_list[i]) 

        ax.plot(self.conf.TARGET_STATE[0],self.conf.TARGET_STATE[1],'b*',markersize=10)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True)
        fig.tight_layout()
        if diff_loc==0:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationSingleInit_{}_{}_torch'.format(self.N_try,n_updates))
        else:
            plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/PolicyEvaluationMultiInit_{}_{}_torch'.format(self.N_try,n_updates))

        plt.clf()
        plt.close(fig)

    def rollout(self,update_step_cntr, actor_model, init_states_sim, diff_loc=0):
        ''' Plot rollout of the actor from some initial states. It generates the results and then calls plot_policy() and plot_policy_eval() '''
        #tau_all_sim = []
        p_ee_all_sim = []

        returns = {}

        for k in range(len(init_states_sim)):
            rollout_controls = np.zeros((self.conf.NSTEPS,self.conf.nb_action))
            rollout_states = np.zeros((self.conf.NSTEPS+1,self.conf.nb_state))
            rollout_p_ee = np.zeros((self.conf.NSTEPS+1,3))
            rollout_episodic_reward = 0

            rollout_p_ee[0,:] = self.env.get_end_effector_position(init_states_sim[k])
            rollout_states[0,:] = np.copy(init_states_sim[k])
            
            for i in range(self.conf.NSTEPS):
                with torch.no_grad():
                    rollout_controls[i, :] = self.NN.eval(actor_model, torch.tensor([rollout_states[i, :]], dtype=torch.float32)).squeeze().numpy()
                #rollout_controls[i,:] = tf.squeeze(self.NN.eval(actor_model, np.array([rollout_states[i,:]]))).numpy()
                rollout_states[i+1,:], rwrd_sim = self.env.step(self.conf.cost_weights_running, rollout_states[i,:],rollout_controls[i,:])
                rollout_p_ee[i+1,:] = self.env.get_end_effector_position(rollout_states[i+1,:])
                
                rollout_p_ee[i+1,-1] = rollout_states[i+1,2] ### !!! ###

                rollout_episodic_reward += rwrd_sim

            if k==0:
                print("N try = {}: Simulation Return @ N updates = {} ==> {}".format(self.N_try,update_step_cntr,rollout_episodic_reward))
                
            p_ee_all_sim.append(rollout_p_ee)  

            returns[init_states_sim[k][0],init_states_sim[k][1]] = rollout_episodic_reward

        self.plot_policy_eval(p_ee_all_sim,update_step_cntr, diff_loc=diff_loc)

        return returns

    def plot_results(self, tau, ee_pos_TO, ee_pos_RL, steps, to=0):
        ''' Plot results from TO and episode to check consistency '''
        timesteps = self.conf.dt*np.arange(steps+1)
        fig = plt.figure(figsize=(12,8))
        if to:
            plt.suptitle('TO EXPLORATION: N try = {}'.format(self.N_try), y=1, fontsize=20)
        else:  
            plt.suptitle('POLICY EXPLORATION: N try = {}'.format(self.N_try), y=1, fontsize=20)

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(timesteps, ee_pos_TO[:,0], 'ro', linewidth=1, markersize=1,legend="x_TO") 
        ax1.plot(timesteps, ee_pos_TO[:,1], 'bo', linewidth=1, markersize=1,legend="y_TO")
        ax1.plot(timesteps, ee_pos_RL[:,0], 'go', linewidth=1, markersize=1,legend="x_RL") 
        ax1.plot(timesteps, ee_pos_RL[:,1], 'ko', linewidth=1, markersize=1,legend="y_RL")
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('[m]')    
        ax1.set_title('End-Effector Position')
        ax1.set_xlim(0, timesteps[-1])
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 2, 3)
        col = ['ro', 'bo', 'go']
        for i in range(self.conf.nb_action):
            ax2.plot(timesteps[:-1], tau[:,i], col[i], linewidth=1, markersize=1,legend='tau{}'.format(i)) 
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Controls')
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(1, 2, 2)
        ax3.plot(ee_pos_TO[:,0], ee_pos_TO[:,1], 'ro', linewidth=1, markersize=2,legend='TO')
        ax3.plot(ee_pos_RL[:,0], ee_pos_RL[:,1], 'bo', linewidth=1, markersize=2,legend='RL')
        ax3.plot([ee_pos_TO[0,0]],[ee_pos_TO[0,1]],'ro',markersize=5)
        ax3.plot([ee_pos_RL[0,0]],[ee_pos_RL[0,1]],'bo',markersize=5)
        obs_plot_list = self.plot_obstaces()
        for i in range(len(obs_plot_list)):
            ax3.add_artist(obs_plot_list[i]) 
        ax3.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
        ax3.set_xlim(self.xlim)
        ax3.set_ylim(self.ylim)
        ax3.set_aspect('equal', 'box')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_title('Plane')
        ax3.legend()
        ax3.grid(True)

        fig.tight_layout()
        #plt.show()

    def plot_Return(self, ep_reward_list):
        ''' Plot returns (not so meaningful given that the initial state, so also the time horizon, of each episode is randomized) '''
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)   
        ax.set_yscale('log') 
        ax.plot(ep_reward_list**2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title("N_try = {}".format(self.N_try))
        ax.grid(True)
        plt.savefig(self.conf.Fig_path+'/N_try_{}'.format(self.N_try)+'/EpReturn_{}_torch'.format(self.N_try))
        plt.close()

    def plot_Critic_Value_function(self, critic_model, n_update, sys_id, name='V'):
        ''' Plot Value function as learned by the critic '''
        if sys_id == 'manipulator':
            N_discretization_x = 60 + 1  
            N_discretization_y = 60 + 1

            plot_data = np.zeros(N_discretization_y*N_discretization_x)*np.nan
            ee_pos = np.zeros((N_discretization_y*N_discretization_x,3))*np.nan

            for k_x in range(N_discretization_x):
                for k_y in range(N_discretization_y):
                    ICS = self.env.reset()
                    ICS[-1] = 0
                    ee_pos[k_x*(N_discretization_y)+k_y,:] = self.env.get_end_effector_position(ICS)
                    plot_data[k_x*(N_discretization_y)+k_y] = self.NN.eval(critic_model, np.array([ICS]))

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot()
            plt.scatter(ee_pos[:,0], ee_pos[:,1], c=plot_data, cmap=cm.coolwarm, antialiased=False)
            obs_plot_list = self.plot_obstaces(a=0.5)
            for i in range(len(obs_plot_list)):
                ax.add_patch(obs_plot_list[i])
            plt.colorbar()
            plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', 'box')
            plt.savefig('{}/N_try_{}/{}_{}_torch'.format(self.conf.Fig_path,self.N_try,name,int(n_update)))
            plt.close()

        else:
            N_discretization_x = 30 + 1  
            N_discretization_y = 30 + 1

            plot_data = np.zeros((N_discretization_y,N_discretization_x))*np.nan

            ee_x = np.linspace(-15, 15, N_discretization_x)
            ee_y = np.linspace(-15, 15, N_discretization_y)

            for k_y in range(N_discretization_y):
                for k_x in range(N_discretization_x):
                    p_ee = np.array([ee_x[k_x], ee_y[k_y], 0])
                    ICS, continue_flag = self.compute_ICS(p_ee, sys_id, continue_flag=0)
                    if continue_flag:
                        continue
                    plot_data[k_x,k_y] = self.NN.eval(critic_model, np.array([ICS]))

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot()
            plt.contourf(ee_x, ee_y, plot_data.T, cmap=cm.coolwarm, antialiased=False)

            obs_plot_list = self.plot_obstaces(a=0.5)
            for i in range(len(obs_plot_list)):
                ax.add_patch(obs_plot_list[i])
            plt.colorbar()
            plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', 'box')
            plt.savefig('{}/N_try_{}/{}_{}'.format(self.conf.Fig_path,self.N_try,name,int(n_update)))
            plt.close()

    def plot_Critic_Value_function_from_sample(self, n_update, NSTEPS_SH, state_arr, reward_arr):
        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        reward_to_go_arr = np.zeros(sum(NSTEPS_SH)+len(NSTEPS_SH)*1)
        idx = 0
        for n in range(len(NSTEPS_SH)):
            for i in range(NSTEPS_SH[n]+1):
                # Compute the partial cost to go
                reward_to_go_arr[idx] = sum(reward_arr[n][i:])
                idx += 1

        state_arr = np.concatenate(state_arr, axis=0)
        ee_pos_arr = np.zeros((len(state_arr),3))
        for i in range(state_arr.shape[0]):
            ee_pos_arr[i,:] = self.env.get_end_effector_position(state_arr[i])
        

        mi = min(reward_to_go_arr)
        ma = max(reward_to_go_arr)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()#projection='3d')
        norm = colors.Normalize(vmin=mi,vmax=ma)

        obs_plot_list = self.plot_obstaces(a=0.5)
        
        ax.scatter(ee_pos_arr[:,0],ee_pos_arr[:,1], c=reward_to_go_arr, norm=norm, cmap=cm.coolwarm, marker='x')
        
        for i in range(len(obs_plot_list)):
            ax.add_patch(obs_plot_list[i])

        plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.coolwarm))
        plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        plt.savefig('{}/N_try_{}/V_sample_{}'.format(self.conf.Fig_path,self.N_try,int(n_update)))
        plt.close()

    def plot_ICS(self,state_arr):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        for j in range(len(state_arr)):
            ax.scatter(state_arr[j][0,0],state_arr[j][0,1])
            obs_plot_list = plot_fun.plot_obstaces()
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
        ax.set_xlim(self.fig_ax_lim[0].tolist())
        ax.set_ylim(self.fig_ax_lim[1].tolist())
        ax.set_aspect('equal', 'box')
        plt.savefig('{}/N_try_{}/ICS_{}_S{}'.format(conf.Fig_path,N_try,update_step_counter,int(w_S)))
        plt.close(fig)

    def plot_rollout_and_traj_from_ICS(self, init_state, n_update, actor_model, TrOp, tag, steps=200):
        ''' Plot results from TO and episode to check consistency '''
        colors = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        
        for j in range(len(init_state)):

            ee_pos_TO = np.zeros((steps,3))
            ee_pos_RL = np.zeros((steps,3))

            RL_states = np.zeros((steps,self.conf.nb_state))
            RL_action = np.zeros((steps-1,self.conf.nb_action))
            RL_states[0,:] = init_state[j,:]
            ee_pos_RL[0,:] = self.env.get_end_effector_position(RL_states[0,:])

            for i in range(steps-1):
                RL_action[i,:] = self.NN.eval(actor_model, torch.tensor(np.array([RL_states[i,:]]), dtype=torch.float32))
                RL_states[i+1,:] = self.env.simulate(RL_states[i,:], RL_action[i,:])
                ee_pos_RL[i+1,:] = self.env.get_end_effector_position(RL_states[i+1,:])
            
            TO_states, _ = TrOp.TO_System_Solve3(init_state[j,:], RL_states.T, RL_action.T, steps-1)

            try:
                for i in range(steps):
                    ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[i,:])
            except:
                ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[0,:])
                
            ax.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax.scatter(ee_pos_TO[0,0],ee_pos_TO[0,1],color=colors[j])
            ax.scatter(ee_pos_RL[0,0],ee_pos_RL[0,1],color=colors[j])
            ax.plot(ee_pos_TO[1:,0],ee_pos_TO[1:,1],color=colors[j])
            ax.plot(ee_pos_RL[1:,0],ee_pos_RL[1:,1],'--',color=colors[j])
        
        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax.add_patch(obs_plot_list[i])

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Plane')
        #ax.legend()
        ax.grid(True)

        plt.savefig('{}/N_try_{}/ee_traj_{}_{}'.format(self.conf.Fig_path,self.N_try,int(n_update), tag))

    def plot_ICS(self, input_arr, cs=0):
        if cs == 1:
            p_arr = np.zeros((len(input_arr),3))
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()
            for j in range(len(input_arr)):
                p_arr[j,:] = input_arr[j,:]
            ax.scatter(p_arr[:,0],p_arr[:,1])
            obs_plot_list = self.plot_obstaces(a = 0.5)
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
            ax.set_xlim(self.conf.fig_ax_lim[0].tolist())
            ax.set_ylim(self.conf.fig_ax_lim[1].tolist())
            ax.set_aspect('equal', 'box')
            ax.grid()
            plt.savefig('{}/N_try_{}/ICS'.format(self.conf.Fig_path,self.N_try))
            plt.close(fig)
        else:    
            p_arr = np.zeros((len(input_arr),3))
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()

            for j in range(len(input_arr)):
                p_arr[j,:] = self.env.get_end_effector_position(input_arr[j])
            ax.scatter(p_arr[:,0],p_arr[:,1])
            obs_plot_list = self.plot_obstaces(a = 0.5)
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
            ax.set_xlim(self.conf.fig_ax_lim[0].tolist())
            ax.set_ylim(self.conf.fig_ax_lim[1].tolist())
            ax.set_aspect('equal', 'box')
            ax.grid()
            plt.savefig('{}/N_try_{}/ICS'.format(self.conf.Fig_path,self.N_try))
            plt.close(fig)

    def plot_traj_from_ICS(self, init_state, TrOp, RLAC, update_step_counter=0,ep=0,steps=200, init=0,continue_flag=1):
        ''' Plot results from TO and episode to check consistency '''
        colors = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        for j in range(len(init_state)):

            ee_pos_TO = np.zeros((steps,3))
            ee_pos_RL = np.zeros((steps,3))
            
            if init == 0:
                # zeros
                _, init_TO_states, init_TO_controls, _, success_init_flag = RLAC.create_TO_init(0, init_state[j,:])
            elif init == 1:
                # NN
                _, init_TO_states, init_TO_controls, _, success_init_flag = RLAC.create_TO_init(1, init_state[j,:])

            if success_init_flag:
                _, _, TO_states, _, _, _  = TrOp.TO_System_Solve(init_state[j,:], init_TO_states, init_TO_controls, steps-1)
            else:
                continue

            try:
                for i in range(steps):
                    ee_pos_RL[i,:] = self.env.get_end_effector_position(init_TO_states[i,:])
                    ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[i,:])
            except:
                ee_pos_RL[i,:] = self.env.get_end_effector_position(init_TO_states[0,:])
                ee_pos_TO[i,:] = self.env.get_end_effector_position(TO_states[0,:])

            ax1.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax1.scatter(ee_pos_RL[0,0],ee_pos_RL[0,1],color=colors[j])
            ax1.plot(ee_pos_RL[1:,0],ee_pos_RL[1:,1],'--',color=colors[j])
                
            ax2.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
            ax2.scatter(ee_pos_TO[0,0],ee_pos_TO[0,1],color=colors[j])
            ax2.plot(ee_pos_TO[1:,0],ee_pos_TO[1:,1],color=colors[j])
        
        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax1.add_patch(obs_plot_list[i])

        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax2.add_patch(obs_plot_list[i])

        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)
        ax1.set_aspect('equal', 'box')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('Warmstart traj.')

        ax2.set_xlim(self.xlim)
        ax2.set_ylim(self.ylim)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X [m]')
        #ax2.set_ylabel('Y [m]')
        ax2.set_title('TO traj.')
        #ax.legend()
        ax1.grid(True)
        ax2.grid(True)

        plt.savefig('{}/N_try_{}/ee_traj_{}_{}_torch'.format(self.conf.Fig_path,self.N_try,init,update_step_counter))

def parse_args():
    ''' Parse the arguments for CACTO training '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test-n',                         type=int,   default=0,                                    
                        help="Test number")
    
    parser.add_argument('--seed',                           type=int,   default=0,                                    
                        help="random and tf.random seed")

    parser.add_argument('--system-id',                      type=str,   default='single_integrator',
                        choices=["single_integrator", "double_integrator", "car", "car_park", "manipulator", "ur5"],
                        help="System-id (single_integrator, double_integrator, car, manipulator, ur5")

    parser.add_argument('--recover-training-flag', action='store_true', help="Flag to recover training")
    
    parser.add_argument('--nb-cpus',                        type=int,   default=10,
                        help="Number of TO problems solved in parallel")
    
    parser.add_argument('--w-S',                            type=float, default=0,
                        help="Sobolev training - weight of the value related error")
    
    ### Testing in progress ###
    parser.add_argument('--GPU-flag', action='store_true', help="Flag to use GPU")
    
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args

if __name__ == '__main__':
    # Get CLI args
    args = parse_args()
    
    # Ignore for now
    GPU_flag = args['GPU_flag']
    N_try     = args['test_n']

    # Set seeds
    if args['seed'] == None:
        random.seed(0)
        seed = random.randint(1,100000)
    else:
        seed = args['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    system_id = args['system_id'] 

    recover_training_flag = args['recover_training_flag']
    
    nb_cpus = args['nb_cpus']

    w_S = args['w_S']

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
        conf_module, env_class, env_TO_class = system_map[system_id]
    except KeyError:
        print('System {} not found'.format(system_id))
        sys.exit()
    
    conf_module, env_class, env_TO_class = system_map[system_id]
    Environment_TO = getattr(importlib.import_module('environment_TO'), env_TO_class) 
    conf = importlib.import_module(conf_module)
    
    # Create folders to store the results and the trained NNs
    for path in conf.path_list:
        os.makedirs(path + '/N_try_{}'.format(N_try), exist_ok=True)
    os.makedirs(conf.Config_path, exist_ok=True)

    # Save configuration
    params = [p for p in conf.__dict__ if not p.startswith("__")]
    with open(conf.Config_path + '/config{}.txt'.format(N_try), 'w') as f:
        for p in params:
            f.write('{} = {}\n'.format(p, conf.__dict__[p]))
        f.write('Seed = {}\n'.format(seed))
        f.write('w_S = {}'.format(w_S))

    shutil.copy('{}.py'.format(conf_module), conf.Config_path + '/' + conf_module + '_{}.py'.format(N_try))
    with open(conf.Config_path + '/' + conf_module + '_{}.py'.format(N_try), 'a') as f:
        f.write('\n\n# {}'.format(args))

    # Create empty txt file in Log_path to store the test info
    open(conf.Log_path + '/info.txt', 'a').close()

    # Common class
    env_TO = Environment_TO

    # 2 instances of each of 4 modified classes + 2
    env_torch = DoubleIntegratortorch(conf)
    env_tf = DoubleIntegratortf(conf)
    
    nn_torch = NN_torch(env_torch, conf)
    nn_tf = NN_tf(env_tf, conf)
    
    rl_tf = RL_AC_tf(env_tf, nn_tf, conf, N_try)
    rl_torch = RL_AC_torch(env_torch, nn_torch, conf, N_try)
    
    plot_fun_torch = PLOT_torch(N_try, env_torch, nn_torch, conf)
    #plot_fun_tf = PLOT_tf(N_try, env_tf, nn_tf, conf)

    TrOp_torch = TO_Casadi(env_torch, conf, env_TO, w_S)
    #TrOp_tf = TO_Casadi(env_tf, conf, env_TO, w_S)

    buffer_torch = ReplayBuffer(conf)
    #buffer_tf = ReplayBuffer_tf(conf)

    rl_tf.setup_model()

    # Get weights of tensorflow models to initialize pytorch models to the same
    weights = []
    for model in (rl_tf.actor_model, rl_tf.critic_model, rl_tf.target_critic):
        tempweights = []
        for layer in model.layers:
            tempweights.append(layer.get_weights())
        weights.append([lst for lst in tempweights if lst])
    
    rl_torch.setup_model(weights=weights)
    #update_step_counter_tf = 0
    update_step_counter_torch = 0
    
    # Save initial weights of the NNs
    #rl_tf.RL_save_weights(update_step_counter_tf)
    rl_torch.RL_save_weights(update_step_counter_torch)

    # Plot initial rollouts
    #plot_fun_tf.plot_traj_from_ICS(np.array(conf.init_states_sim), TrOp_tf, rl_tf, update_step_counter=update_step_counter_tf,steps=conf.NSTEPS, init=0)
    plot_fun_torch.plot_traj_from_ICS(np.array(conf.init_states_sim), TrOp_torch, rl_torch, update_step_counter=update_step_counter_torch,steps=conf.NSTEPS, init=0)

    # Initialize arrays to store the reward history of each episode and the average reward history of last 100 episodes
    ep_arr_idx = 0
    ep_reward_arr = np.zeros(conf.NEPISODES-ep_arr_idx)*np.nan  

    def compute_sample(args):
            ''' Create samples solving TO problems starting from given ICS '''
            ep = args[0]
            ICS = args[1]

            # Create initial TO #
            init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH, success_init_flag = rl_torch.create_TO_init(ep, ICS)
            if success_init_flag == 0:
                return None
                
            # Solve TO problem #
            TO_controls, TO_states, success_flag, TO_ee_pos_arr, TO_step_cost, dVdx = TrOp_torch.TO_Solve(init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)
            if success_flag == 0:
                return None
            
            # Collect experiences 
            state_arr, partial_reward_to_go_arr, total_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr  = rl_torch.RL_Solve(TO_controls, TO_states, TO_step_cost)

            if conf.env_RL == 0:
                RL_ee_pos_arr = TO_ee_pos_arr

            return NSTEPS_SH, TO_controls, TO_ee_pos_arr, dVdx, state_arr.tolist(), partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr

    def create_unif_TO_init(n_UICS=1):
        ''' Create n uniformely distributed ICS '''
        # Create ICS TO #
        init_rand_state = env_torch.reset()
        
        return init_rand_state

    time_start = time.time()

    ### START TRAINING ###
    print(f'Training start')

    for ep in range(conf.NLOOPS): 
        # Generate and store conf.EP_UPDATE random-uniform ICS
        tmp = []
        init_rand_state = []
        for i in range(conf.EP_UPDATE):
            init_rand_state.append(create_unif_TO_init(i))

        for i in range(conf.EP_UPDATE):
            result = compute_sample((ep, init_rand_state[i]))
            tmp.append(result)

        # Remove unsuccessful TO problems and update EP_UPDATE
        tmp = [x for x in tmp if x is not None]
        NSTEPS_SH, TO_controls, ee_pos_arr_TO, dVdx, state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, ee_pos_arr_RL = zip(*tmp)

        # Update the buffer
        buffer_torch.add(state_arr, partial_reward_to_go_arr, state_next_rollout_arr, dVdx, done_arr, term_arr)
        #buffer_tf.add(state_arr, partial_reward_to_go_arr, state_next_rollout_arr, dVdx, done_arr, term_arr)  

        # Update NNs, reseeding before each update so that the sampling is consistent between both calls
        temp_seed = random.randint(0, 1000)
        np.random.seed(temp_seed)
        #update_step_counter_tf = rl_tf.learn_and_update(update_step_counter_tf, buffer_tf, ep)
        np.random.seed(temp_seed)
        update_step_counter_torch = rl_torch.learn_and_update(update_step_counter_torch, buffer_torch, ep)

        #assert update_step_counter_torch == update_step_counter_tf
        # plot Critic value function
        #plot_fun.plot_Critic_Value_function(RLAC.critic_model, update_step_counter, system_id) ###

        # Plot rollouts and state and control trajectories
        if update_step_counter_torch%conf.plot_rollout_interval_diff_loc == 0 or system_id == 'single_integrator' or system_id == 'double_integrator' or system_id == 'car_park' or system_id == 'car' or system_id == 'manipulator':
            print("System: {} - N_try = {}".format(conf.system_id, N_try))
            #plot_fun_tf.plot_Critic_Value_function(rl_tf.critic_model, update_step_counter_tf, system_id)
            plot_fun_torch.plot_Critic_Value_function(rl_torch.critic_model, update_step_counter_torch, system_id)
            #plot_fun_tf.plot_traj_from_ICS(np.array(conf.init_states_sim), TrOp_tf, rl_tf, update_step_counter=update_step_counter_tf, ep=ep,steps=conf.NSTEPS, init=1)
            plot_fun_torch.plot_traj_from_ICS(np.array(conf.init_states_sim), TrOp_torch, rl_torch, update_step_counter=update_step_counter_torch, ep=ep,steps=conf.NSTEPS, init=1)

        # Update arrays to store the reward history and its average
        ep_reward_arr[ep_arr_idx:ep_arr_idx+len(tmp)] = ep_return
        ep_arr_idx += len(tmp)

        for i in range(len(tmp)):
            print("Episode  {}  --->   Return = {}".format(ep*len(tmp) + i, ep_return[i]))

        if update_step_counter_torch > conf.NUPDATES:
            break

    time_end = time.time()
    print('Elapsed time: ', time_end-time_start)

    # Plot returns
    #plot_fun_tf.plot_Return(ep_reward_arr)
    plot_fun_torch.plot_Return(ep_reward_arr)

    # Save networks at the end of the training
    #rl_tf.RL_save_weights()
    rl_torch.RL_save_weights()

    # Simulate the final policy
    #plot_fun_tf.rollout(update_step_counter_tf, rl_tf.actor_model, conf.init_states_sim)
    plot_fun_torch.rollout(update_step_counter_torch, rl_torch.actor_model, conf.init_states_sim)


    
       