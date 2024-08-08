import numpy as np
import random
import unittest
import numpy as np
import tensorflow as tf
import torch
import uuid
import math
import importlib
import numpy as np
import torch.nn as nn
from utils import normalize_tensor
from TO import TO_Casadi
import math
import random
import numpy as np
import tensorflow as tf
from keras import layers, regularizers, initializers
from tf_siren import SinusodialRepresentationDense
import pinocchio as pin
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from utils import normalize_tensor

# Import configuration file and environment file
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



def reset_pt():
    ''' Choose initial state uniformly at random '''
    state = np.zeros(conf.nb_state)

    time = np.random.uniform(conf.x_init_min[-1], conf.x_init_max[-1])
    for i in range(conf.nb_state-1): 
        state[i] = np.random.uniform(conf.x_init_min[i], conf.x_init_max[i])
    state[-1] = conf.dt*round(time/conf.dt)

    return state

def reset_tf():
    ''' Choose initial state uniformly at random '''
    state = np.zeros(conf.nb_state)

    time = random.uniform(conf.x_init_min[-1], conf.x_init_max[-1])
    for i in range(conf.nb_state-1): 
        state[i] = random.uniform(conf.x_init_min[i], conf.x_init_max[i]) 
    state[-1] = conf.dt*round(time/conf.dt)

    return state

print(reset_pt())
print(reset_tf())
