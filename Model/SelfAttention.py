import torch
import numpy as np
import math
import torch.nn as nn

# Source: https://github.com/ok1zjf/VASNet/blob/master/vasnet_model.py
class SelfAttention(nn.Module):

    def __init__(self,input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()


        self.m = input_size
        self.output_size = output_size
        self.dhidden = 1024
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)



    def forward(self, x):
        n = x.shape[0]  # sequence length
        
        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)

        return y, att_weights_


#Source : https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    
def getPositionEncoding(seq_len, d, n=100000):
    pe = torch.zeros(seq_len, d)    
    # create position column   
    k = torch.arange(0, seq_len).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d, 2) * -(math.log(n) / d)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)
    pe = pe.unsqueeze(0)
  
    return pe
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class VASNet(nn.Module):

    def __init__(self,depth=1,dim=1024,pos_enc=False):
        super(VASNet, self).__init__()

        self.m = 1024 # cnn features size
        self.hidden_size = dim
        self.depth = depth
        self.att = SelfAttention(input_size = self.m,output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        self.pos_enc = pos_enc

    def forward(self, x):
        if len(x.shape)>2:
            m = x.shape[2] # Feature size
            # Place the video frames to the batch dimension to allow for batch arithm. operations.
            # Assumes input batch size = 1.
            seq_len = x.shape[1]
        else:
            m = x.shape[1]
            seq_len = x.shape[0]
        if self.pos_enc:
            x_pos = getPositionEncoding(seq_len,self.hidden_size)
            x = x + x_pos.to(x.device)
        x = x.view(-1, m)
        y ,_ = self.att(x)
        #y = y + x
        y= y+ x 
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y        
    
# An adapted version of the Self Attention model that incorporates self attention    
class VASNetPC(nn.Module):

    def __init__(self,depth=1,dim=1024,pos_enc=True,**kwargs):
        super(VASNetPC, self).__init__()

        self.m = 1024 # cnn features size
        self.hidden_size = dim
        self.depth = depth
        self.att = SelfAttention(input_size = self.m,output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        self.pos_enc = pos_enc

    def forward(self, x):

        m = x.shape[2] # Feature size
        # Place the video frames to the batch dimension to allow for batch arithm. operations.
        # Assumes input batch size = 1.
        seq_len = x.shape[1]
        if self.pos_enc:
            x_pos = getPositionEncoding(seq_len,self.hidden_size)
            x = x + x_pos.to(x.device)
        x = x.view(-1, m)
        y ,_ = self.att(x)
        #y = y + x
        y= y+ x 
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y        
#-------------------------------------------------------------------- MLP--------------------------------------------------------------------------------------------
class MLPM(nn.Module):
    def __init__(self,dim=1024):
        super(MLPM,self).__init__()
        self.m = 1024 # cnn features size
        self.hidden_size = dim
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.sig = nn.Sigmoid()

    def forward(self, x):

            # Just giving the inputs directly to the regressor
            y =x
            # Frame level importance score regression
            # Two layer NN
            y = self.ka(y)
            y = self.relu(y)
            y = self.drop50(y)
            y = self.layer_norm_ka(y)

            y = self.kd(y)
            y = self.sig(y)
            y = y.view(1, -1)

            return y