import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
class CriticModelTF(tf.keras.Model):
    def __init__(self):
        super(CriticModelTF, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Ones())
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Ones())
        self.dense3 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Ones())

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

class CriticModelTorch(torch.nn.Module):
    def __init__(self):
        super(CriticModelTorch, self).__init__()
        self.dense1 = nn.Linear(4, 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 1)
        torch.nn.init.constant_(self.dense1.weight, 1)
        torch.nn.init.constant_(self.dense2.weight, 1)
        torch.nn.init.constant_(self.dense3.weight, 1)
        torch.nn.init.constant_(self.dense1.bias, 1)
        torch.nn.init.constant_(self.dense2.bias, 1)
        torch.nn.init.constant_(self.dense3.bias, 1)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        return self.dense3(x)

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
def compute_critic_grad_tf(critic_model, target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch):
        ''' Compute the gradient of the critic NN '''
        MSEtf = tf.keras.losses.MeanSquaredError()
        #print(f'{state_batch.shape}, {state_next_rollout_batch.shape}, {partial_reward_to_go_batch.shape}, {dVdx_batch.shape}, {d_batch.shape}, {weights_batch.shape}')
        with tf.GradientTape() as tape: 
            # Compute value function tail if TD(n) is used
            if dummy:
                reward_to_go_batch = partial_reward_to_go_batch
            else:     
                target_values = target_critic(state_next_rollout_batch)                                 # Compute Value at next state after conf.nsteps_TD_N steps given by target critic                 
                reward_to_go_batch = partial_reward_to_go_batch + (1-d_batch)*target_values                        # Compute batch of 1-step targets for the critic loss                    
            
            # Compute critic loss
            if w_S != 0:
                with tf.GradientTape() as tape2:
                    tape2.watch(state_batch)                  
                    critic_value = critic_model(state_batch)
                der_critic_value = tape2.gradient(critic_value, state_batch)
                #print(f'der_critic_value_tf: {der_critic_value}')   
                
                critic_loss_v = MSEtf(reward_to_go_batch, critic_value, sample_weight=weights_batch)
                #print(f'critic_loss_v_tf: {critic_loss_v}')
                vxlog =  custom_logarithm_tf(dVdx_batch[:, :-1])
                #print(f'vxlog_tf: {vxlog}')
                derlog = custom_logarithm_tf(der_critic_value[:, :-1])
                #print(f'vxlog_tf.shape: {vxlog.shape}')
                #print(f'derlog_tf.shape: {derlog.shape}')
                #print(f'weights_tf.shape: {weights_batch.shape}')
                critic_loss_der = MSEtf(vxlog, derlog, sample_weight=weights_batch) # dV/dt not computed and so not used in the update

                critic_loss = critic_loss_der + w_S*critic_loss_v
            else:
                critic_value = critic_model(state_batch)
                critic_loss = MSEtf(reward_to_go_batch, critic_value, sample_weight=weights_batch)

        # Compute the gradients of the critic loss w.r.t. critic's parameters
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)   

        return critic_grad, reward_to_go_batch, critic_value, target_critic(state_batch)

def compute_critic_grad_torch(critic_model, target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch):
        critic_model.train()
        target_critic.eval()
        MSEtorch=WeightedMSELoss()
        othermse = torch.nn.MSELoss()

        reward_to_go_batch = partial_reward_to_go_batch if dummy else partial_reward_to_go_batch + (1 - d_batch) * self.eval(target_critic, state_next_rollout_batch)

        critic_model.zero_grad()

        if w_S != 0:
            state_batch.requires_grad_(True)
            critic_value = critic_model(state_batch)
            der_critic_value = torch.autograd.grad(outputs=critic_value, inputs=state_batch, grad_outputs=torch.ones_like(critic_value), create_graph=True)[0]
            critic_loss_v = MSEtorch(reward_to_go_batch, critic_value, weights=weights_batch)

            vxlog =  custom_logarithm_torch(dVdx_batch[:, :-1])
            derlog = custom_logarithm_torch(der_critic_value[:, :-1])
            #print(f'vxlog_torch.shape: {vxlog.shape}')
            #print(f'derlog_torch.shape: {derlog.shape}')
            #print(f'weights_torch.shape: {weights_batch.shape}')
            critic_loss_der = MSEtorch(vxlog, derlog, weights=weights_batch)
            critic_loss = critic_loss_der + w_S * critic_loss_v
        else:
            critic_value = critic_model(state_batch)
            critic_loss = MSEtorch(reward_to_go_batch, critic_value, weight=weights_batch)

        #critic_grad = torch.autograd.grad(critic_loss, critic_model.parameters())
        critic_model.zero_grad()
        #print(mean_Qneg)
        #actor_grad = torch.autograd.grad(mean_Qneg, actor_model.parameters())
        #actor_model.zero_grad()
        critic_loss.backward()
        critic_grad = [param.grad for param in critic_model.parameters()]

        return critic_grad, reward_to_go_batch, critic_value, target_critic(state_batch)



np.random.seed(100)
torch.manual_seed(100)
tf.random.set_seed(100)
# Initialize models
critic_model_tf = CriticModelTF()
target_critic_tf = CriticModelTF()

critic_model_torch = CriticModelTorch()
target_critic_torch = CriticModelTorch()

# Generate test data
state_batch_np = np.random.rand(32, 4).astype(np.float32)
state_next_rollout_batch_np = np.random.rand(32, 4).astype(np.float32)
partial_reward_to_go_batch_np = np.random.rand(32, 1).astype(np.float32)
dVdx_batch_np = np.random.rand(32, 4).astype(np.float32)
d_batch_np = np.random.rand(32, 1).astype(np.float32)
weights_batch_np = np.random.rand(32, 1).astype(np.float32)

state_batch_tf = tf.convert_to_tensor(state_batch_np)
state_next_rollout_batch_tf = tf.convert_to_tensor(state_next_rollout_batch_np)
partial_reward_to_go_batch_tf = tf.convert_to_tensor(partial_reward_to_go_batch_np)
dVdx_batch_tf = tf.convert_to_tensor(dVdx_batch_np)
d_batch_tf = tf.convert_to_tensor(d_batch_np)
weights_batch_tf = tf.convert_to_tensor(weights_batch_np)

state_batch_torch = torch.tensor(state_batch_np, requires_grad=True)
state_next_rollout_batch_torch = torch.tensor(state_next_rollout_batch_np)
partial_reward_to_go_batch_torch = torch.tensor(partial_reward_to_go_batch_np)
dVdx_batch_torch = torch.tensor(dVdx_batch_np)
d_batch_torch = torch.tensor(d_batch_np)
weights_batch_torch = torch.tensor(weights_batch_np)


critic_grad_tf, reward_to_go_batch_tf, critic_value_tf, target_critic_value_tf = compute_critic_grad_tf(critic_model_tf, target_critic_tf, state_batch_tf, state_next_rollout_batch_tf, partial_reward_to_go_batch_tf, dVdx_batch_tf, d_batch_tf, weights_batch_tf)

critic_grad_torch, reward_to_go_batch_torch, critic_value_torch, target_critic_value_torch = compute_critic_grad_torch(critic_model_torch, target_critic_torch, state_batch_torch, state_next_rollout_batch_torch, partial_reward_to_go_batch_torch, dVdx_batch_torch, d_batch_torch, weights_batch_torch)

# Compare results
def compare_tensors(tensor_tf, tensor_torch):
    tensor_torch_np = tensor_torch.detach().numpy() if isinstance(tensor_torch, torch.Tensor) else np.array(tensor_torch)
    return np.allclose(tensor_tf.numpy(), tensor_torch_np, atol=1e-5)

print("Reward to Go Batch Comparison:", compare_tensors(reward_to_go_batch_tf, reward_to_go_batch_torch))
print("Critic Value Comparison:", compare_tensors(critic_value_tf, critic_value_torch))
print("Target Critic Value Comparison:", compare_tensors(target_critic_value_tf, target_critic_value_torch))

# Compare gradients
print(critic_grad_tf)
print('-'*40)
print(critic_grad_torch)
for grad_tf, grad_torch in zip(critic_grad_tf, critic_grad_torch):
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
