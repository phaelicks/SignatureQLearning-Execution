import numpy as np
import signatory
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigQFunction(nn.Module):
    def __init__(self, env, sig_depth, in_channels=None): 
        assert (
            env.observation_space.shape[1] == 1
        ), "Observation space variables must be scalars"        
        
        super().__init__()
        if in_channels == None:
            self.in_channels = env.observation_space.shape[0] # w/o action
        else:
            self.in_channels = in_channels
        self.out_dimension = env.action_space.n
        #self.num_actions = env.action_space.n
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

    def compute_signature(self, path, basepoint=False):
        if path.shape[1] == 1 and not basepoint:
            return signatory.signature(path=path, depth=self.sig_depth,
                                       basepoint=path.squeeze(0))  
        else:
            return signatory.signature(path=path, depth=self.sig_depth,
                                       basepoint=basepoint)      


    def update_signature(self, new_path, basepoint, signature):
        """
        This function updates a given signature with new data from a path.
        Let S be the signature of a path X and Y and new path with Y[0] = X[-1],
        then it returns the signature of the concatenation of X and Y.

        Arguments:
            - new_path:     data from a new path Y given as a 3-d tensor of shape 
                            (batch, length, in_channels)
            - basepoint:    the last value of the path X from which signature was computed,
                            a 3-d tensor of shape (batch, 1, in_channels)
            - signature:    the signature from a previous path X, given as a 2-d 
                            tensor of shape (batch, self.sig_channels)

        Returns:
            - signature of X and Y concatenated, a 2-d tensor of shape (batch, self.sig_channels)
        """

        return signatory.signature(path=new_path, depth=self.sig_depth,
                                   basepoint=basepoint, initial=signature) 


    def initialize_parameters(self, uniform=None, factor=None, zero_bias=True):
        # weights
        if uniform != None and factor != None:
            raise ValueError("Choose either uniform or multiplicative factor.")
        elif uniform != None:
            self.linear.weight.data.uniform_(-uniform, uniform)
            nn.init.xavier_uniform_(self.linear.weight)
        elif factor != None:
            self.linear.weight.data *= factor
        # bias
        if zero_bias == True:
            self.linear.bias.data.fill_(0)
        elif zero_bias == False and factor != None:
            self.linear.bias.data *= factor


class RNNPolicy(nn.Module):
    def __init__(self, env, layers=1, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = env.observation_space.shape[0] # w/o action
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
    
class LSTMPolicy(nn.Module):
    def __init__(self, env, layers=1, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = env.observation_space.shape[0] # w/o action
        self.out_dimension = env.action_space.n
    
        self.lstm = nn.LSTM(self.in_channels, 32, layers, batch_first=True)
        # specify hidden layers when initialzed
        self.fc1 = nn.Linear(32, self.out_dimension)
        
    def forward(self, seq):
        # seq here is the observation-action history
        out, _ = self.lstm(seq)
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
