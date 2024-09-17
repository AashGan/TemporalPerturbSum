import torch
import torch.nn as nn
from .SelfAttention import getPositionEncoding # Absolute positional encoding
import math
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self,input_dims,transformer_dims,feedforward_dims,heads,masking=None,**kwargs):
        super(TransformerBlock,self).__init__()
        self.attn = MultiHeadSelfAttention(input_dims, transformer_dims, heads,**kwargs)
        dropout = kwargs.get('dropout',0.5)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dims, feedforward_dims),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(feedforward_dims, input_dims)
        )
        self.norm1 = nn.LayerNorm(input_dims)
        self.mask_hype = masking
        self.droplayer = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(input_dims)
        self.skip_lin = kwargs.get('skip_linear',False)
    @staticmethod
    def create_local_mask(seq_len,local_width,rand_percentage = 0.2):
        local_width = min(local_width,int(seq_len))
        mask = torch.zeros((seq_len,seq_len))
        

        for i in range(seq_len):
                mask[i,max(0,i-local_width//2):min(seq_len,i+local_width//2+1)] =1 
        random_mask = torch.bernoulli(rand_percentage*torch.ones_like(mask))
        mask = torch.logical_or(mask,random_mask)
        return mask
    def forward(self, x, mask=None):
        if self.mask_hype:
            if not self.training:
            
                mask  = TransformerBlock.create_local_mask(x.shape[1],self.mask_hype[0],self.mask_hype[1])
            else:
                mask  = TransformerBlock.create_local_mask(x.shape[1],self.mask_hype[0],self.mask_hype[1])
        attn_out = self.attn(x, mask=mask)
        if type(attn_out) is tuple:
            attn_out = attn_out[0]
        x = x + self.droplayer(attn_out)
        x = self.norm1(x)
        if  not self.skip_lin:
            linear_out = self.linear_net(x)
            x = x + self.droplayer(linear_out)
            x = self.norm2(x)
        return x
    
class TransformerSum(nn.Module):
    def __init__(self,input_dims,transformer_dims,feedforward_dims,depth,heads,mask_params,**kwargs):
          super(TransformerSum,self).__init__()
          assert len(heads) == depth and len(mask_params)==depth, "Make sure the transformer has the parameter defined for each depth"
          pos_enc = kwargs.pop('pos_enc',[None]*depth)
          self.att_blocks = nn.ModuleList([TransformerBlock(input_dims,transformer_dims,feedforward_dims,heads[i],masking=mask_params[i],pos_enc=pos_enc[i],**kwargs) for i in range(depth)])
          self.predictor =  nn.Sequential(nn.Linear(input_dims,1),nn.Sigmoid())
    def forward(self,x):
        for l in self.att_blocks:
             x = l(x)
        return self.predictor(x)
        

          
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,input_size,output_size=1024,heads=8,**kwargs):
        super(MultiHeadSelfAttention,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.heads = heads
        self.head_dim = int(output_size/heads)
        self.pos_enc = kwargs.get('pos_enc',False)
        self.freq = 10000
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)
        self.q = nn.Linear(input_size,output_size)
        self.k = nn.Linear(input_size,output_size)
        self.v = nn.Linear(input_size,output_size)
        self.out = nn.Linear(output_size,input_size)
        self.return_att = kwargs.get('return_att',False)
        self.skip_att = kwargs.get('skip_att',False)
        self.scale_param = nn.Parameter(torch.tensor(math.sqrt(self.head_dim)),requires_grad=kwargs.get('enable_scale',False))
   
    def change_for_mul(self,vector):
        
        return vector.view(vector.shape[0],self.heads,vector.shape[1],self.head_dim)
    

    
    def forward(self,x,mask=None):
        # Space for positional Encoding
        if self.skip_att:
            return x
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        

        if self.pos_enc == "absolute":
            query += getPositionEncoding(query.shape[1],query.shape[2]).to(query.device)
            key += getPositionEncoding(key.shape[1],key.shape[2]).to(key.device)
        
        query = self.change_for_mul(query)
        key = self.change_for_mul(key)
        value = self.change_for_mul(value)

        attention_scores = torch.matmul(query/self.scale_param, key.transpose(-1, -2))

        if mask is not None:
            if mask.device != attention_scores.device:
                mask = mask.to(attention_scores.device)
            attention_scores = attention_scores.masked_fill(mask==0,-9e15)
        attention_scores = F.softmax(attention_scores,dim=-1)

        final_values = torch.matmul(attention_scores,value)

        outputs = self.out(final_values.view(final_values.shape[0],final_values.shape[2],self.head_dim*self.heads))
        if self.return_att:
            return outputs,attention_scores
        return outputs

