# Used for initialization of pytorch networks
import tensorflow as tf
from keras import layers, regularizers
from tf_siren import SinusodialRepresentationDense

class TFACNetworks:
    '''
    Simplified actor-critic networks class used to initialize pytorch models 
    '''
    def __init__(self, conf):
        '''    
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
    '''
        self.conf = conf

        critic_funcs = {
            'elu': self.create_critic_elu,
            'sine': self.create_critic_sine,
            'sine-elu': self.create_critic_sine_elu,
            'relu': self.create_critic_relu
        }

        self.actor_model = self.create_actor()
        self.critic_model = critic_funcs[self.conf.critic_type]()
        self.target_critic = critic_funcs[self.conf.critic_type]()
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

    def listWeights(self):
        '''
        Returns list of weights of trainable layers in all 3 models. 
        '''
        weights = []
        for model in (self.actor_model, self.critic_model, self.target_critic):
            temp_weights = []
            for layer in model.layers:
                temp_weights.append(layer.get_weights())
            weights.append([x for x in temp_weights if x])
        return weights




