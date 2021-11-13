import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadBIRUAttention(nn.Module):

    def __init__(self, bert, batch_size, hidden_units, embed_dim, num_heads, embedding_length, output_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.bert = bert
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_size = batch_size
        self.embedding_length = embedding_length
        self.output_size = output_size
        self.hidden_units = hidden_units 
        self.multihead = nn.MultiheadAttention(self.embedding_length , num_heads)
        self.gru = nn.GRU(64, 32, bidirectional=True, batch_first=True)
        self.label = nn.Linear(32, self.output_size)
        self.qkv_proj = nn.Linear(self.embedding_length, 3*self.hidden_units)
        self.o_proj = nn.Linear(self.hidden_units, 64)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def initialize_hidden_state(self, device):
        return torch.zeros((2, self.batch_size, 32)).to(device)

    def forward(self, x, mask=None, return_attention=False, batch_size = None):
        with torch.no_grad():
            input = self.bert(x)[0]
        if batch_size is None:
            self.hidden = Variable(torch.zeros(2, self.batch_size, 32).cuda())
        else:
            self.hidden = Variable(torch.zeros(2, batch_size, 32).cuda())
            
        batch_size, seq_length, embed_dim = input.size()
        qkv = self.qkv_proj(input)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, 100)
        attn_output = self.o_proj(values)
        output, self.hidden  = self.gru(attn_output, self.hidden)
        outp = self.label(self.hidden[-1])
        return F.log_softmax(outp, dim=-1)
        
     