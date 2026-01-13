import torch

class RoPEEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=self.register_buffer('theta',None)
        self.cos_freq=self.register_buffer('cos_freq',None)
        self.sin_freq=self.register_buffer('sin_freq',None)
        
    def forward(self,x): # x: [batch_size,head,seq_len,hidden_size]
        seq_len,hidden_size=x.size(-2),x.size(-1)
        
        # lazy init
        if self.theta is None:
            self.theta=torch.pow(10000,-torch.arange(0,hidden_size,2,dtype=x.dtype,device=x.device)/hidden_size).view(1,1,1,hidden_size//2) # [1,1,1,hidden_size//2]
        
        # cos&sin cache
        if self.cos_freq is None or seq_len>self.cos_freq.size(2):
            m=torch.arange(0,seq_len,dtype=x.dtype,device=x.device).view(1,1,seq_len,1) # [1,1,seq_len,1]
            freqs=m*self.theta
            cos_freq=torch.cos(freqs) # (1,1,seq_len,hidden_size//2)
            sin_freq=torch.sin(freqs) # (1,1,seq_len,hidden_size//2)
            self.cos_freq=torch.concat([cos_freq,cos_freq],dim=-1) # (1,1,seq_len,hidden_size)
            self.sin_freq=torch.concat([sin_freq,sin_freq],dim=-1) # (1,1,seq_len,hidden_size)
        
        x1=torch.concat((-x[...,hidden_size//2:],x[...,:hidden_size//2]),dim=-1)
        return x*self.cos_freq[:seq_len]+x1*self.sin_freq[:seq_len]

if __name__=='__main__':
    q=torch.arange(0,1*1*5*20).reshape(1,1,5,20)

    rope=RoPEEmbedding()
    q_embed=rope(q)
    print(q,'\n',q_embed)
    
    q=torch.arange(0,1*1*10*20).reshape(1,1,10,20)
    q_embed=rope(q)
    print(q,'\n',q_embed)