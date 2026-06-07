import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as f

class CustomLargeLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, n_layer):
        super().__init__()
        self.block_size = block_size
        # 1. Token Embedding: Maps token integer IDs to a vector space of size 'd_model'
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        # 2. Position Embedding: Tells the model *where* the token sits in the sequence
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        # 3. Transformer Blocks: A stack of 'n_layer' transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_head= 6) for _ in range(n_layer)])\
        # 4. Layer Normalization: Normalizes the output of the transformer blocks
        self.ln_f = nn.LayerNorm(d_model)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Retrieve token vectors -> Shape: (B, T, C)
        tok_emb = self.token_embedding_table(idx)
        #Create position indices and retrieve position vectors -> Shape: (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        # Combine token and position embeddings -> Shape: (B, T, C)
        x = tok_emb + pos_emb
        
        #Pass through transformer blocks -> Shape: (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)

        #Project back to vocab dimension to get raw scores (logits)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Reshape logits and targets for loss computation
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            #Compute cross-entropy loss
            loss = f.cross_entropy(logits, targets)

        return logits, loss

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        #Communication Layer: Multi-Head Self-Attention
        self.sa = MultiHeadAttention(n_head, d_model // n_head, d_model)
        #Computation Layer
        self.ffwd = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # LayerNorm -> Communication -> Add Residual
        x = x + self.sa(self.ln1(x))
        # LayerNorm -> Computation -> Add Residual
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, d_model):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size

        #Linear projections to create Queries, Keys and Values Simultaneously
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model)

        # The Causal Mask: Lower triangular matrix filled with 1s, upper with 0s
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)))
    
    def forward(self, x):
        B, T, C = x.shape

        # 1. Linear projection to Q, K, V in one shot, then split into heads
        qkv = self.qkv_proj(x) # Shape: (B * T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Rearrange shapes for multi-


