import tensorflow as tf
from keras import layers, regularizers, initializers
import numpy as np

def wreg():
    state_input = layers.Input(shape=(16,))
    outputs = layers.Dense(16, activation='elu', kernel_initializer=initializers.Ones(),
                           bias_initializer=initializers.Ones(), 
                           kernel_regularizer=regularizers.l1_l2(1e-2,1e-2),
                           bias_regularizer=regularizers.l1_l2(1e-2,1e-2))(state_input)
    return tf.keras.Model(inputs=state_input, outputs=outputs)

def noreg():
    state_input = layers.Input(shape=(16,))
    outputs = layers.Dense(16, activation='elu', kernel_initializer=initializers.Ones(),
                           bias_initializer=initializers.Ones())(state_input)
    return tf.keras.Model(inputs=state_input, outputs=outputs)

yes = wreg()
no = noreg()
inn = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])  # Added batch dimension
print(yes(inn))
print(no(inn))
