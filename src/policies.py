import signatory
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SigPolicy(nn.Module):
    def __init__(self, env, sig_depth): 
        assert (
            env.observation_space.shape[1] == 1
        ), "Observation space variables must be scalars"        
        
        super().__init__()

        self.in_channels = env.observation_space.shape[0] + 1 # add action
        self.out_dimension = 1
        self.num_actions = env.action_space.n
        self.sig_depth = sig_depth
        self.sig_channels = signatory.signature_channels(channels=self.in_channels,
                                                         depth=sig_depth)
        self.linear = torch.nn.Linear(self.sig_channels, self.out_dimension, bias=True)
        #self.linear1 = torch.nn.Linear(self.sig_channels, 32, bias=True)
        #self.linear2 = torch.nn.Linear(32, out_dimension, bias = True)


    def forward(self, signature):
        """ 
        The signature of a path fed through a single linear layer.
        :signature: is a two dimensional tensor of shape (batch, self.sig_channels). 
        Returns a two dimensional tensor of shape (batch, out_dimension).      
        """
        #x = self.linear1(signature)
        #x = F.normalize(x)
        #x = self.linear2(x)
        #x = F.normalize(signature)
        return self.linear(signature)

    def update_signature(self, path, basepoint=None, signature=None, remove=False):
        """
        - This function updates a given :signature: with new stream :path: where 
        :basepoint: is the last value of the old path from which :signature: was computed
        - If remove == True it instead returns the signatures of the old path, 
        shortend by :path: at the left end. In this case :path: must be a subpath of old path
        - If only :path: is given, it returns its signature, which in case length of
        path is 1 is a tensor of zeros with shape (1, self.sig_channels)
            
            :path: is a three dimensional tensor of shape (batch, length, in_channels)
            :basepoint: is a three dimentional tensor of shape (batch, 1, in_channels)
            :signature: is a two dimensional tensor of shape (batch, self.sig_channels)
        """
        if basepoint==None and signature==None:
            if path.shape[1] == 1: # only one observation, return zeros
                return signatory.signature(path, depth=self.sig_depth,
                                           basepoint = path.squeeze(0)) # alternatively set basepoint=True                         
                                           #basepoint=torch.tensor([[-0.5, 1.]],dtype=torch.float)) 
            else: # return new signature 
                return signatory.signature(path, depth=self.sig_depth)
        elif remove: # return signature of path shortened by :path:
                # TODO: check if this functionality does as intendet
                path = torch.flip(path, [0, 1]) # reverse path
                sig_redundant = signature.signature(path, depth=self.sig_depth)
                return signatory.signature_combine(sig_redundant, signature, 
                                                   path.shape[2], self.sig_depth) 
        else: # update signature
            return signatory.signature(path, depth=self.sig_depth,
                                       basepoint=basepoint, initial=signature) 

    def create_Q_values(self, signature, last_tuple_tensor, new_observation):
        Q = []
        for action in range(self.num_actions):
            new_tuple = np.hstack((new_observation, action))
            new_tuple_tensor = torch.tensor(
                [new_tuple], requires_grad=False, dtype=torch.float
            )
            new_path = torch.cat((last_tuple_tensor, new_tuple_tensor), 0).unsqueeze(0)
            signature_1 = self.update_signature(new_path, last_tuple_tensor, signature)
            action_value = self.forward(signature_1)[0] # unwrap from batch dimension
            Q.append(action_value)
        return torch.cat(Q, 0)
        
            


    def initialize_parameters(self, uniform=None, factor=None, zero_bias=True):
        # weights
        if uniform != None and factor != None:
            raise RuntimeError("Choose either uniform or multiplicative factor.")
        elif uniform != None:
            self.linear.weight.data.uniform_(-uniform, uniform)
            nn.init.xavier_uniform_(self.linear.weight)
        elif factor != None:
            self.linear.weigh.data *= factor
        # bias
        if zero_bias == True:
            self.linear.bias.data.fill_(0)
        if zero_bias == False and factor != None:
            self.linear.bias.data *= factor

class RNNPolicy(nn.Module):
    def __init__(self, env, layers=1, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = env.observation_space.n + 1 # one action
        self.out_dimension = env.action_space.n
    
        self.rnn = nn.RNN(self.in_channels, 32, layers, nonlinearity="relu", batch_first=True)
        # specify hidden layers when initialzed
        self.fc1 = nn.Linear(32, self.out_dimension)
        
    def forward(self, seq):
        # seq here is the observation-action history
        out, _ = self.rnn(seq)
        out = out[:, -1, :]
        out = self.fc1(out)
        return out
    
class RandomPolicy(nn.Module):
    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.action_space = env.action_space.n
        
    def forward(self, seq):  
        r = np.random.randint(0, self.action_space)
        return torch.eye(self.action_space)[r] # probability of chossen action to one
