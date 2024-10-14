import torch
import numpy as np

class GPUBuffer:
    def __init__(self, conf, device=torch.device('cuda:0')):
        '''
        :input conf :                           (Configuration file)

            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param nb_state :                   (int) State size (robot state size + 1)
        '''

        self.conf = conf
        self.device = device
        self.storage_mat = torch.zeros((conf.REPLAY_SIZE, conf.nb_state + 1 + conf.nb_state + conf.nb_state + 1 + 1), device=device)
        self.next_idx = 0
        self.full = 0
        self.exp_counter = torch.zeros(conf.REPLAY_SIZE, device=self.device)
    
    def concatenate_sample(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
        ''' Convert batch of transitions into a tensor '''
        obses_t = np.concatenate(obses_t, axis=0)
        rewards = np.concatenate(rewards, axis=0)                                 
        obses_t1 = np.concatenate(obses_t1, axis=0)
        dVdxs = np.concatenate(dVdxs, axis=0)
        dones = np.concatenate(dones, axis=0)
        terms = np.concatenate(terms, axis=0)
        
        np_sample = np.concatenate((obses_t, rewards.reshape(-1,1), obses_t1, dVdxs, dones.reshape(-1,1), terms.reshape(-1,1)),axis=1)

        return torch.tensor(np_sample, device=self.device)
       
    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dVdxs, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        obses_t = torch.tensor(obses_t, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)                                  
        obses_t1 = torch.tensor(obses_t1, dtype=torch.float32)
        dVdxs = torch.tensor(dVdxs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        return obses_t, rewards, obses_t1, dVdxs, dones, weights
    
    # def concatenate_sample(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
    #     ''' Convert batch of transitions into a tensor '''
    #     obses_t = torch.cat(torch.tensor(obses_t, device=self.device),dim=0)
    #     rewards = torch.cat(torch.tensor(rewards, device=self.device), dim=0)
    #     obses_t1 = torch.cat(torch.tensor(obses_t1, device=self.device), dim=0)
    #     dVdxs = torch.cat(torch.tensor(dVdxs, device=self.device), dim=0)
    #     dones = torch.cat(torch.tensor(dones, device=self.device), dim=0)
    #     terms = torch.cat(torch.tensor(terms, device=self.device), dim=0)
    #     print(obses_t.data_ptr())
    #     rewards = rewards.view(-1, 1)  # reshapes to column vector
    #     dones = dones.view(-1, 1)
    #     terms = terms.view(-1, 1)

    #     # Concatenate everything along the 1st axis (like in the NumPy code)
    #     return torch.cat((obses_t, rewards, obses_t1, dVdxs, dones, terms), dim=1)
    
    def add(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
        ''' Add transitions to the buffer '''
        data = self.concatenate_sample(obses_t, rewards, obses_t1, dVdxs, dones, terms)
        #data = self.convert_sample_to_tensor(*data)

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
        idxes = torch.randint(0, max_idx, size=(self.conf.BATCH_SIZE,)) 

        obses_t = self.storage_mat[idxes, :self.conf.nb_state]
        rewards = self.storage_mat[idxes, self.conf.nb_state:self.conf.nb_state+1]
        obses_t1 = self.storage_mat[idxes, self.conf.nb_state+1:self.conf.nb_state*2+1]
        dVdxs = self.storage_mat[idxes, self.conf.nb_state*2+1:self.conf.nb_state*3+1]
        dones = self.storage_mat[idxes, self.conf.nb_state*3+1:self.conf.nb_state*3+2]
        terms = self.storage_mat[idxes, self.conf.nb_state*3+2:self.conf.nb_state*3+3]

        # Priorities not used
        weights = torch.ones((self.conf.BATCH_SIZE,1), device=self.device)
        batch_idxes = None
        
        return obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, batch_idxes