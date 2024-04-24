import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimensions, num_heads):
        super(MultiHeadAttention, self).__init__

        # Check dimension model is divisible by num_heads
        assert model_dimensions % num_heads == 0

        # Initialize Model
        self.dimensions = model_dimensions # Dimensions of Model
        self.num_heads = num_heads # Attention Heads
        self.d_k = model_dimensions // num_heads # Dimensions for Q, K, and V

        # Layers for outputs
        self.W_q = nn.Linear(in_features=self.dimensions, out_features=self.dimensions) # Query
        self.W_k = nn.Linear(in_features=self.dimensions, out_features=self.dimensions) # Key
        self.W_v = nn.Linear(in_features=self.dimensions, out_features=self.dimensions) # Value 
        self.W_o = nn.Linear(in_features=self.dimensions, out_features=self.dimensions) # Output
    

    # Scaled Dot Product Attention
    def sdp_attention(self, Q, K, V, mask=None):

        # Attention formula = Softmax(Q * K^t + Mask) * V

        # Calculate Attention Scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Mask attention from Padding/sequences after current
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 
        attn_probabilities = torch.matmul(attn_scores, dim=-1)

        # Return Matrix Multiplication of Softmax and Value matrix
        return torch.matmul(attn_probabilities, V)

    def combine_heads(self, x):
        
        # Combine heads to original shape
        batch_size, _, seq_length, d_k = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dimensions)
    
    def split_heads(self, x):

        # Get num_heads for MultiHeadAttention
        batch_size, seq_length, dimensions = x.size()

        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):

        # Apply linear transformation and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Scaled Dot Product Attention
        attn_output = self.sdp_attention(Q, K, V, mask)

        # Return Transformation on Combined heads
        return self.W_o(self.combine_heads(attn_output))
    

class FeedForwardPositionWise(nn.Module):
    def __init__(self, model_dimensions, ff_dimensions):
        super(FeedForwardPositionWise, self).__init__()
        self.fc1 = nn.Linear(model_dimensions, ff_dimensions)
        self.fc2 = nn.Linear(ff_dimensions, model_dimensions)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, model_dimensions, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # Positional Encodings
        pe = torch.zeros(max_seq_length, model_dimensions)
        # Position indices for each position in sequence
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Term for scaling position indices
        div_term = torch.exp(torch.arange(0, model_dimensions, 2).float() * -(math.log(10000.0) / model_dimensions))
        
        # Sin applied to even indices, Cos applied to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer (non trainable part of module state)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):

        # Add positional encodings to x (x.size(1) checks that positional encodings match sequence length of x)
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, model_dimensions, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(model_dimensions, num_heads)
        self.feed_forward = FeedForwardPositionWise(model_dimensions, d_ff)
        
        self.norm1 = nn.LayerNorm(model_dimensions)
        self.norm2 = nn.LayerNorm(model_dimensions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
    
    