import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
class WeightedMSELoss(nn.Module):
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
        mse_loss = torch.pow(inputs - targets, 2)
        
        if weights is not None:
            #print(inputs.shape)
            #print(weights.shape)
            if weights.shape != inputs.shape:
                raise ValueError("Weights must have the same shape as inputs and targets")
            mse_loss = mse_loss * weights
        
        return mse_loss  # Return the mean of the loss
# Define the same model structure for both TensorFlow and PyTorch
class CriticModelTF(tf.keras.Model):
    def __init__(self):
        super(CriticModelTF, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

class CriticModelTorch(nn.Module):
    def __init__(self):
        super(CriticModelTorch, self).__init__()
        self.dense1 = nn.Linear(4, 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        return self.dense3(x)

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

# Define custom logarithm function for TensorFlow
def custom_logarithm_tf(input):
    positive_log = tf.math.log(tf.math.maximum(input, 1e-7) + 1)
    negative_log = -tf.math.log(tf.math.maximum(-input, 1e-7) + 1)
    result = tf.where(input > 0, positive_log, negative_log)
    return result

# Define compute_critic_grad function for TensorFlow
def compute_critic_grad_tf(critic_model, target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch):
    with tf.GradientTape() as tape:
        reward_to_go_batch = partial_reward_to_go_batch if conf.MC else partial_reward_to_go_batch + (1 - d_batch) * target_critic(state_next_rollout_batch)
        if w_S != 0:
            with tf.GradientTape() as tape2:
                tape2.watch(state_batch)
                critic_value = critic_model(state_batch)
            der_critic_value = tape2.gradient(critic_value, state_batch)
            critic_loss_v = tf.reduce_mean(tf.keras.losses.mean_squared_error(reward_to_go_batch, critic_value) * weights_batch)
            critic_loss_der = tf.reduce_mean(tf.keras.losses.mean_squared_error(custom_logarithm_tf(dVdx_batch[:, :-1]), custom_logarithm_tf(der_critic_value[:, :-1])) * weights_batch)
            critic_loss = critic_loss_der + w_S * critic_loss_v
        else:
            critic_value = critic_model(state_batch)
            critic_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(reward_to_go_batch, critic_value) * weights_batch)
    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    return critic_grad, reward_to_go_batch, critic_value, target_critic(state_batch)

# Define custom logarithm function for PyTorch
def custom_logarithm_torch(input):
    positive_log = torch.log(torch.maximum(input, torch.tensor(1e-7)) + 1)
    negative_log = -torch.log(torch.maximum(-input, torch.tensor(1e-7)) + 1)
    result = torch.where(input > 0, positive_log, negative_log)
    return result

# CustomGradient class for PyTorch
class CustomGradient:
    def __init__(self, conf, MSE, w_S):
        self.conf = conf
        self.MSE = MSE
        self.w_S = w_S

    def eval(self, model, inputs):
        model.eval()
        with torch.no_grad():
            return model(inputs)

    def compute_critic_grad(self, critic_model, target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch):
        critic_model.train()
        target_critic.eval()

        reward_to_go_batch = partial_reward_to_go_batch if self.conf.MC else partial_reward_to_go_batch + (1 - d_batch) * self.eval(target_critic, state_next_rollout_batch)

        critic_model.zero_grad()

        if self.w_S != 0:
            state_batch.requires_grad_(True)
            critic_value = critic_model(state_batch)
            der_critic_value = torch.autograd.grad(outputs=critic_value, inputs=state_batch, grad_outputs=torch.ones_like(critic_value), create_graph=True)[0]
            print(reward_to_go_batch.shape)
            print(critic_value.shape)
            print(weights_batch.shape)
            critic_loss_v = self.MSE(reward_to_go_batch, critic_value, weights=weights_batch)
            critic_loss_der = self.MSE(custom_logarithm_torch(dVdx_batch[:, :-1]), custom_logarithm_torch(der_critic_value[:, :-1]), weights=weights_batch)
            critic_loss = critic_loss_der + self.w_S * critic_loss_v
        else:
            critic_value = critic_model(state_batch)
            critic_loss = self.MSE(reward_to_go_batch, critic_value, weight=weights_batch)

        critic_loss.backward()
        critic_grad = [p.grad for p in critic_model.parameters()]

        return critic_grad, reward_to_go_batch, critic_value, self.eval(target_critic, state_batch)

# Initialize CustomGradient class for PyTorch
conf = type('conf', (object,), {"MC": False})  # Dummy configuration
w_S = 0.5
MSE = WeightedMSELoss()

custom_gradient = CustomGradient(conf, MSE, w_S)

# Run both implementations
critic_grad_tf, reward_to_go_batch_tf, critic_value_tf, target_critic_value_tf = compute_critic_grad_tf(critic_model_tf, target_critic_tf, state_batch_tf, state_next_rollout_batch_tf, partial_reward_to_go_batch_tf, dVdx_batch_tf, d_batch_tf, weights_batch_tf)

critic_grad_torch, reward_to_go_batch_torch, critic_value_torch, target_critic_value_torch = custom_gradient.compute_critic_grad(critic_model_torch, target_critic_torch, state_batch_torch, state_next_rollout_batch_torch, partial_reward_to_go_batch_torch, dVdx_batch_torch, d_batch_torch, weights_batch_torch)

# Compare results
def compare_tensors(tensor_tf, tensor_torch):
    tensor_torch_np = tensor_torch.detach().numpy() if isinstance(tensor_torch, torch.Tensor) else np.array(tensor_torch)
    return np.allclose(tensor_tf.numpy(), tensor_torch_np, atol=1e-5)

print("Reward to Go Batch Comparison:", compare_tensors(reward_to_go_batch_tf, reward_to_go_batch_torch))
print("Critic Value Comparison:", compare_tensors(critic_value_tf, critic_value_torch))
print("Target Critic Value Comparison:", compare_tensors(target_critic_value_tf, target_critic_value_torch))

# Compare gradients
for grad_tf, grad_torch in zip(critic_grad_tf, critic_grad_torch):
    print("Gradient Comparison:", compare_tensors(grad_tf, grad_torch))
