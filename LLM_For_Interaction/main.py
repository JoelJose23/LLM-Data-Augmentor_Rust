import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as f

class CustomLargeLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, n_layer):
        """
        The master container for the Decoder-Only Transformer.
        
        Args:
            vocab_size (int): Total size of your dictionary (set by your Rust tokenizer).
            d_model (int): Hidden dimension / size of the vector space representing one token.
            block_size (int): The Context Window (maximum sequence length the model can process).
            n_layer (int): Total number of Transformer blocks to stack vertically.
        """
        super().__init__()
        self.block_size = block_size
        
        # 1. Token Embedding: Acts as a lookup table mapping integer token IDs to dense vectors.
        # Table Shape: (vocab_size, d_model)
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        
        # 2. Position Embedding: Holds learned vectors that describe token indices (0, 1, 2... up to block_size).
        # This solves the problem of Transformers being permutation-invariant (having no inherent sense of word order).
        # Table Shape: (block_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        
        # 3. Transformer Blocks: Stacking blocks sequentially. Python unpacks the list comprehension using '*'.
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_head=6) for _ in range(n_layer)])
        
        # 4. Final Layer Normalization: Standardizes the outputs across channels before the final projection.
        self.ln_f = nn.LayerNorm(d_model)
        
        # 5. Language Model Head: The final linear classifier projecting hidden vectors back to vocabulary space.
        # Linear Projection Shape: (d_model, vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        Executes the network's forward pass.
        
        Args:
            idx (Tensor): Current input integers of shape (B, T), where B = Batch size, T = Sequence length.
            targets (Tensor, optional): Correct next-token labels of shape (B, T) used for cross-entropy loss.
        """
        B, T = idx.shape
        
        # Extract token semantics from table -> Yields shape: (B, T, C) where C = d_model channels
        tok_emb = self.token_embedding_table(idx)
        
        # Create an integer sequence from 0 to T-1 on the same hardware device (CPU or GPU) as the data,
        # then lookup position vectors -> Yields shape: (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        # Combine semantics and position. PyTorch automatically "broadcasts" the (T, C) tensor
        # across the (B) batch dimension to fit perfectly -> Shape remains: (B, T, C)
        x = tok_emb + pos_emb
        
        # Funnel the combined vector matrix through the stack of Transformer blocks -> Shape: (B, T, C)
        x = self.blocks(x)
        
        # Normalize the structural channels -> Shape: (B, T, C)
        x = self.ln_f(x)

        # Scale hidden layers back into token predictions -> Yields raw scores (logits) of shape: (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # OPTIMIZATION: PyTorch's cross_entropy requires a flat 2D input for predictions (N, C)
            # and a flat 1D array for labels (N). We break down our 3D tensors into a flat batch-sequence timeline.
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # Flattens out batch and time: (B*T, vocab_size)
            targets = targets.view(B*T)   # Flattens target indices: (B*T)
            
            # Cross entropy performs Softmax automatically under the hood, calculates log probabilities,
            # and penalizes the network based on how low it scored the true target token.
            loss = f.cross_entropy(logits, targets)

        return logits, loss

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head):
        """
        A single structural unit combining Communication (Attention) and Computation (FeedForward).
        Uses a Pre-LayerNorm configuration (standard in modern GPT architectures).
        """
        super().__init__()
        # Ensure the hidden dimension splits evenly across your attention heads
        self.sa = MultiHeadAttention(n_head, d_model // n_head, d_model)
        self.ffwd = FeedForward(d_model)
        
        # Independent layer norm modules for isolation
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 🛣️ RESIDUAL HIGHWAY 1: Normalize -> Self-Attend -> Add the pristine original 'x' back.
        # This addition creates a direct path for raw mathematical gradients to flow backward
        # during training without shrinking or exploding.
        x = x + self.sa(self.ln1(x))
        
        # 🛣️ RESIDUAL HIGHWAY 2: Normalize -> Run isolated Feed-Forward MLPs -> Add 'x' back.
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, d_model):
        """
        Processes multi-head causal attention so tokens can query past context.
        """
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size

        # Rather than constructing 3 separate linear layers for Query, Key, and Value, we bundle them
        # into a single large matrix projection for massive GPU speedups.
        # Output size is 3 * d_model so it holds all packed vectors together.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model)

        # 🧱 THE CAUSAL MASK: A static matrix buffer filled with 1s on/below diagonal, and 0s above.
        # It is registered as a "buffer" so PyTorch tracks it as state but doesn't calculate gradients for it.
        # Dimension is locked at a generous 1024x1024 to accommodate large context limits.
        self.register_buffer("tril", torch.tril(torch.ones(1024, 1024)))
    
    def forward(self, x):
        B, T, C = x.shape

        # 1. Fire the combined projection layer -> Yields tensor shape: (B, T, 3*C)
        qkv = self.qkv_proj(x)
        
        # Split the large 3*C dimension cleanly into 3 isolated tensors: Queries, Keys, and Values.
        # Each chunk inherits shape: (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # MULTI-HEAD SHUFFLE: We reshape the single 'C' dimension into (n_head, head_size).
        # Then, we .transpose(1, 2) to move the 'n_head' axis up front.
        # This lets PyTorch compute all attention heads in parallel using fast matrix math.
        # Target parallel shape for all tensors: (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 2. SCALED DOT-PRODUCT ATTENTION: Multiply Queries by Transposed Keys to evaluate relational affinities.
        # Scaled down by 1 / sqrt(head_size) to prevent the dot products from growing massive
        # in higher dimensions, which would flatten the Softmax gradients.
        # Matrix Multiply Shape: (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        weights = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)

        # 3. CAUSAL ENFORCEMENT: Slice the pre-registered mask to fit the current sequence length T.
        # Anywhere the mask matrix contains a zero, overwrite the affinity score to -Infinity.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax turns the scores into a clean probability distribution along the last row axis.
        # When Softmax evaluates -inf, it transforms into absolute 0.0 probability—completely 
        # erasing the model's ability to "look ahead" into the target future.
        weights = f.softmax(weights, dim=-1) # Shape: (B, n_head, T, T)

        # 4. CONTEXT GATHERING: Multiply the attention heatmaps by the Value vectors.
        # Tokens dynamically extract numeric features from lines they evaluated high affinity for.
        # Matrix Multiply Shape: (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        out = weights @ v 

        # REASSEMBLE HEADS: Transpose back to (B, T, n_head, head_size).
        # .contiguous() re-arranges the raw memory addresses so they are sequential, allowing us to
        # .view() flatten the heads back down into the original unified channel dimension 'C'.
        # Shape collapses back smoothly to standard parameters: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final linear projection mix before returning to the block layer
        return self.out_projection(out)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        """
        The isolated thinking engine. After attention manages communication,
        the multi-layer perceptron (MLP) applies non-linear computations to each token vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            # Standard practice expands hidden dimensions 4 times larger to allocate model capacity
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(), # Gaussian Error Linear Unit: smooth, modern non-linearity used in GPT models
            nn.Linear(4 * d_model, d_model) # Shrink back down to baseline channels
        )
    def forward(self, x):
        # Operates identically and independently on every individual token position vector
        return self.net(x) # Shape remains completely unchanged: (B, T, C)