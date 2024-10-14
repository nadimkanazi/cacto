import numpy as np
import torch
import torch.nn as nn
from siren_pytorch import Siren
from utils import normalize_tensor

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

class NN:
    def __init__(self, env, conf, w_S=0):
        '''    
        :input env :                            (Environment instance)

        :input self.conf :                           (self.configuration file)

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
    
    def weightCopy(self, model, weights):
        '''
        Copies the given weights from the TF model into the pytorch model
        '''
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
        return model
    
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
            model = self.weightCopy(model, weights)
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        return model.to(torch.float32)

    def create_critic_elu(self, weights=None): 
        ''' Create critic NN - elu'''
        #Tested Successfully#
        model = nn.Sequential(
            nn.Linear(self.conf.nb_state, 16),
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
            model = self.weightCopy(model, weights)
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        return model.to(torch.float32)
    
    def create_critic_sine_elu(self, weights=None): 
        ''' Create critic NN - elu'''
        #Tested Successfully#
        model = nn.Sequential(
            Siren(self.conf.nb_state, 64),
            nn.Linear(64, 64),
            nn.ELU(),
            Siren(64, 128),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128,1)
        )
        if weights is not None:
            model = self.weightCopy(model, weights)
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        return model.to(torch.float32)
        
    def create_critic_sine(self, weights=None): 
        ''' Create critic NN - elu'''
        model = nn.Sequential(
            Siren(self.conf.nb_state, 64),
            Siren(64, 64),
            Siren(64, 128),
            Siren(128, 128),
            nn.Linear(128, 1)
        )
        if weights is not None:
            model = self.weightCopy(model, weights)
        else:
            nn.init.xavier_uniform_(model[-1].weight)
            nn.init.constant_(model[-1].bias, 0)
        return model.to(torch.float32)
        
    def create_critic_relu(self, weights=None): 
        ''' Create critic NN - relu'''
        #Tested Successfully#
        model = nn.Sequential(
            nn.Linear(self.conf.nb_state, 16),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(16, 32),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(32, self.conf.NH1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(self.conf.NH1, self.conf.NH2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(self.conf.NH2, 1)
        )
        if weights is not None:
            model = self.weightCopy(model, weights)
        else:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        return model.to(torch.float32)
    
    def eval(self, NN, input):
        ''' Compute the output of a NN given an input '''
        #Tested Successfully#
        if not torch.is_tensor(input):
            if isinstance(input, list):
                input = np.array(input)
            input = torch.tensor(input, dtype=torch.float32)

        if self.conf.NORMALIZE_INPUTS:
            input = normalize_tensor(input, torch.tensor(self.conf.state_norm_arr, dtype=torch.float32))

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
        reward_to_go_batch = partial_reward_to_go_batch if self.conf.MC else partial_reward_to_go_batch + (1 - d_batch) * self.eval(target_critic, state_next_rollout_batch)

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
        
        total_loss = critic_loss #+ self.compute_reg_loss(critic_model, False)
        critic_model.zero_grad()
        total_loss.backward()

        return reward_to_go_batch, critic_value, self.eval(target_critic, state_batch)

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
            batch_size = self.conf.BATCH_SIZE

        actions = self.eval(actor_model, state_batch)

        # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
        act_np = actions.detach().cpu().numpy()
        
        state_next_tf, ds_next_da = self.env.simulate_batch(state_batch.detach().cpu().numpy(), act_np), self.env.derivative_batch(state_batch.detach().cpu().numpy(), act_np)
        #state_next_tf = torch.rand(size=(128,5), requires_grad=True)
        #ds_next_da = torch.rand(size=(128,5,2), requires_grad=True)
        #print(state_next_tf.shape)
        #print(ds_next_da.shape)
        
        
        #state_next_tf = state_next_tf.clone().detach().to(dtype=torch.float32).requires_grad_(True)
        #ds_next_da = ds_next_da.clone().detach().to(dtype=torch.float32).requires_grad_(True)

        # Compute critic value at the next state
        critic_value_next = self.eval(critic_model, state_next_tf)

        # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)
        dV_ds_next = torch.autograd.grad(outputs=critic_value_next, inputs=state_next_tf,
                                        grad_outputs=torch.ones_like(critic_value_next),
                                        create_graph=True)[0]

        cost_weights_terminal_reshaped = torch.tensor(self.conf.cost_weights_terminal, dtype=torch.float32).reshape(1, -1)
        cost_weights_running_reshaped = torch.tensor(self.conf.cost_weights_running, dtype=torch.float32).reshape(1, -1)

        # Compute rewards
        state_batch_np = state_batch.detach().cpu().numpy()
        
        temp1 = torch.matmul(torch.tensor(term_batch, dtype=torch.float32), cost_weights_terminal_reshaped)
        
        temp2 = torch.matmul(torch.tensor(1 - term_batch, dtype=torch.float32), cost_weights_running_reshaped)
        
        rewards_tf = self.env.reward_batch(temp1 + temp2, state_batch_np, actions)
        #print(rewards_tf.shape)
        # dr_da = gradient of reward r(s,a) w.r.t. policy's action a
        dr_da = torch.autograd.grad(outputs=rewards_tf, inputs=actions,
                                    grad_outputs=torch.ones_like(rewards_tf),
                                    create_graph=True)[0]

        dr_da_reshaped = dr_da.view(batch_size, 1, self.conf.nb_action)

        # dr_ds' + dV_ds' (note: dr_ds' = 0)
        dQ_ds_next = dV_ds_next.view(batch_size, 1, self.conf.nb_state)

        # (dr_ds' + dV_ds')*ds'_da
        dQ_ds_next_da = torch.bmm(dQ_ds_next, ds_next_da)

        # (dr_ds' + dV_ds')*ds'_da + dr_da
        dQ_da = dQ_ds_next_da + dr_da_reshaped

        # Multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a
        actions = self.eval(actor_model, state_batch)
        actions_reshaped = actions.view(batch_size, self.conf.nb_action, 1)
        dQ_da_reshaped = dQ_da.view(batch_size, 1, self.conf.nb_action)
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
        #NOTE: layers in the original tf code were using kreg_l2_C (from self.conf) for all regularization parameters. 
        #This doesn't make sense and was changed here. Also, the original codebase used the keras 
        #bias_regularizer and kernel_regularizer variables, but never accessed the actor_model.losses
        #parameter to actually use the regularization loss in gradient computations.
        #I ended up not using this since it caused issues
        reg_loss = 0
        kreg_l1 = 0
        kreg_l2 = 0
        breg_l1 = 0
        breg_l2 = 0
        if actor:
            kreg_l1 = self.conf.kreg_l1_A
            kreg_l2 = self.conf.kreg_l2_A
            breg_l1 = self.conf.breg_l1_A
            breg_l2 = self.conf.breg_l2_A
        else:
            kreg_l1 = self.conf.kreg_l1_C
            kreg_l2 = self.conf.kreg_l2_C
            breg_l1 = self.conf.breg_l1_C
            breg_l2 = self.conf.breg_l2_C

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
