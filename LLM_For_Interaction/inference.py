import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# 1. MODEL ARCHITECTURE (Must match training exactly)
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, max_seq_len: int = 4096):
        super().__init__()
        self.head_size = head_size
        inv_freq = 1.0 / (10_000 ** (torch.arange(0, head_size, 2).float() / head_size))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.cos_cache = None
        self.sin_cache = None
        self._build_cache(max_seq_len, "cpu")

    def _build_cache(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cache = emb.cos()
        self.sin_cache = emb.sin()
        self.max_seq_len_cached = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_start_pos: int):
        seq_len = q.shape[1]
        device = q.device
        needed_len = seq_start_pos + seq_len

        if self.cos_cache is None or needed_len > self.max_seq_len_cached or self.cos_cache.device != device:
            alloc_len = max(needed_len, self.max_seq_len_cached * 2)
            self._build_cache(alloc_len, device)
            
        positions = torch.arange(seq_start_pos, seq_start_pos + seq_len, device=device)
        cos = self.cos_cache[positions].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cache[positions].unsqueeze(0).unsqueeze(2)

        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (self._rotate_half(k) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.head_size = d_model // n_head
        self.attn_drop = attn_dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_size)

    def forward(self, x: torch.Tensor, past_kv=None, seq_start_pos: int = 0, max_cache_len: int = 2048):
        B, S_q, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = q.view(B, S_q, self.n_head, self.head_size)
        k = k.view(B, S_q, self.n_head, self.head_size)
        v = v.view(B, S_q, self.n_head, self.head_size)

        q, k = self.rope(q, k, seq_start_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            if k.size(-2) > max_cache_len + S_q:
                k = k[:, :, -(max_cache_len + S_q):, :]
                v = v[:, :, -(max_cache_len + S_q):, :]

        current_kv = (k.detach(), v.detach())
        S_k = k.size(-2)

        if past_kv is not None:
            past_len = S_k - S_q
            mask = torch.ones(S_q, S_k, dtype=torch.bool, device=x.device)
            mask = torch.tril(mask, diagonal=past_len)
            mask = mask.unsqueeze(0).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        out = out.transpose(1, 2).contiguous().view(B, S_q, C)
        return self.out_projection(out), current_kv

class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(4 * d_model, d_model),
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_dropout: float = 0.0, ffn_dropout: float = 0.0):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, n_head, attn_dropout)
        self.ffwd = FeedForward(d_model, ffn_dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, past_kv=None, seq_start_pos: int = 0):
        attn_out, current_kv = self.sa(self.ln1(x), past_kv, seq_start_pos)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, current_kv

class CustomLargeLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int = 6):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, target_ids=None, past_kv=None, seq_start_pos=0):
        x = self.token_embedding(input_ids)
        current_kv_list = []
        for i, block in enumerate(self.blocks):
            block_past_kv = past_kv[i] if past_kv is not None else None
            x, block_kv = block(x, block_past_kv, seq_start_pos)
            current_kv_list.append(block_kv)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None, current_kv_list

# ==========================================
# 2. RUST INTERACTION DECOUPLED TOKENS
# ==========================================
RUST_URL = "http://127.0.0.1:3000"

def encode_via_rust(text: str) -> list[int]:
    """Sends raw text to Rust web server and receives token integer IDs."""
    try:
        response = requests.post(f"{RUST_URL}/encode", json={"text_content": text})
        response.raise_for_status()
        return response.json()["input_ids"] # Adjust key based on Rust API schema
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to encode via Rust server. Is it running? Details: {e}")

def decode_via_rust(token_ids: list[int]) -> str:
    """Sends predicted token IDs back to Rust to be decoded into string segments."""
    try:
        response = requests.post(f"{RUST_URL}/decode", json={"tokens": token_ids})
        response.raise_for_status()
        return response.json()["text_content"] # Adjust key based on Rust API schema
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to decode via Rust server. Is it running? Details: {e}")

# ==========================================
# 3. KV-CACHE GENERATION LOGIC
# ==========================================
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=0.7, top_k=8, repetition_penalty=1.2, device="cuda"):
    model.eval()
    
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_tokens = []
    
    # Token IDs for the <|im_end|> sequence (character-level: < | i m _ e n d | >)
    IM_END_IDS = [85, 84, 9, 13, 70, 5, 14, 4, 84, 86]
    
    seq_start_pos = 0
    logits, _, past_kv = model(x, past_kv=None, seq_start_pos=seq_start_pos)
    next_token_logits = logits[:, -1, :] 
    
    for _ in range(max_new_tokens):
        next_token_logits = next_token_logits / max(temperature, 1e-5)
        
        # Apply repetition penalty to already-generated tokens
        if repetition_penalty != 1.0 and generated_tokens:
            for prev_id in set(generated_tokens):
                if next_token_logits[0, prev_id] > 0:
                    next_token_logits[0, prev_id] /= repetition_penalty
                else:
                    next_token_logits[0, prev_id] *= repetition_penalty
        
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        next_id = next_token.item()
        generated_tokens.append(next_id)
        
        # Stop if model produced the <|im_end|> sequence
        if len(generated_tokens) >= len(IM_END_IDS) and generated_tokens[-len(IM_END_IDS):] == IM_END_IDS:
            break
        
        # Stream decode back to screen token-by-token for that clean LLM typing look
        print(decode_via_rust([next_id]), end="", flush=True)
        
        seq_start_pos += x.size(1) 
        x = next_token 
        
        logits, _, past_kv = model(x, past_kv=past_kv, seq_start_pos=seq_start_pos)
        next_token_logits = logits[:, -1, :]
        
    return generated_tokens

# ==========================================
# 4. RUNTIME MAIN
# ==========================================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "checkpoints/ckpt_epoch_2_doc_230000.pt" 
    
    VOCAB_SIZE = 98
    D_MODEL = 384
    N_LAYER = 6
    N_HEAD = 6

    print(f"Initializing architecture weights onto {DEVICE}...")
    model = CustomLargeLanguageModel(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL,
        n_layer=N_LAYER, n_head=N_HEAD
    ).to(DEVICE)
    
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    print("Checkpoint structural configurations validated and loaded successfully.")

    # Format using your structural ChatML sequence parameters
    user_query = "How are you feeling today?"
    formatted_prompt = f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\nProcessing prompt through Rust Server Tokenizer...")
    prompt_ids = encode_via_rust(formatted_prompt)
    
    print("\n" + "="*40)
    print(f"PROMPT: {user_query}")
    print("="*40)
    print("RESPONSE: ", end="")

    # Fire off autoregressive cache generation loop
    generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=256,
        temperature=0.2,
        top_k=5,
        device=DEVICE
    )
    print("\n" + "="*40)

if __name__ == "__main__":
    main()