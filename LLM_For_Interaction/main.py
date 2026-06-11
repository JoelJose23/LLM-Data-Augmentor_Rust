import os
import time
import math
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. LAZY-LOADING BINARY DATASET (SINGLE DOC SCHEMA)
# ==========================================
class DocumentDataset(Dataset):
    MAGIC = b"TOKN"

    def __init__(self, bin_path: str):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary not found: {bin_path}.")

        self.bin_path = bin_path
        self.offsets = []
        self.lengths = []

        print(f"Indexing {bin_path} structural layout for sequential reading...")
        with open(bin_path, "rb") as f:
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError(f"Bad magic header: expected TOKN, got {magic!r}")

            raw_n_entries = f.read(4)
            (n_entries,) = struct.unpack("<I", raw_n_entries)

            for i in range(n_entries):
                offset = f.tell()
                (length,) = struct.unpack("<I", f.read(4))
                f.seek(8 * length, 1)  # Skip inputs and targets
                self.offsets.append(offset)
                self.lengths.append(length)

        print(f"  Successfully indexed {len(self.offsets):,} full documents. Max Seq: {max(self.lengths)}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        offset = self.offsets[idx]
        with open(self.bin_path, "rb") as f:
            f.seek(offset)
            (length,) = struct.unpack("<I", f.read(4))
            inp_bytes = f.read(length * 4)
            tgt_bytes = f.read(length * 4)

        inp = torch.frombuffer(bytearray(inp_bytes), dtype=torch.int32).to(torch.long).clone()
        tgt = torch.frombuffer(bytearray(tgt_bytes), dtype=torch.int32).to(torch.long).clone()
        return inp, tgt

# ==========================================
# 2. ROTARY POSITIONAL EMBEDDINGS (OFFSET AWARE)
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        device = q.device

        needed_len = seq_start_pos + seq_len

        if self.cos_cache is None or needed_len > self.max_seq_len_cached or self.cos_cache.device != device:
            alloc_len = max(needed_len, self.max_seq_len_cached * 2)
            self._build_cache(alloc_len, device)
            
        positions = torch.arange(seq_start_pos, seq_start_pos + seq_len, device=device)
        cos = self.cos_cache[positions].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cache[positions].unsqueeze(0).unsqueeze(2)

        return (q * cos) + (self._rotate_half(q) * sin), \
               (k * cos) + (self._rotate_half(k) * sin)

# ==========================================
# 3. CONTEXT-AWARE ATTENTION (WITH KV-CACHE)
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
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
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop if self.training else 0.0
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.attn_drop if self.training else 0.0
            )

        out = out.transpose(1, 2).contiguous().view(B, S_q, C)
        return self.out_projection(out), current_kv

# ==========================================
# 4. FEED-FORWARD & TRANSFORMER BLOCKS
# ==========================================
class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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

# ==========================================
# 5. LANGUAGE MODEL
# ==========================================
class CustomLargeLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int = 6, attn_dropout: float = 0.0, ffn_dropout: float = 0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, attn_dropout, ffn_dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None, past_kv: list | None = None, seq_start_pos: int = 0):
        x = self.token_embedding(input_ids)
        current_kv_list = []
        
        for i, block in enumerate(self.blocks):
            block_past_kv = past_kv[i] if past_kv is not None else None
            x, block_kv = block(x, block_past_kv, seq_start_pos)
            current_kv_list.append(block_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)

        return logits, loss, current_kv_list

# ==========================================
# 6. LIVE METRICS TRACKER
# ==========================================
class TrainingTracker:
    def __init__(self, save_dir="metrics_plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.doc_indices = []
        self.raw_losses = []
        self.smoothed_losses = []
        self.docs_per_sec = []
        self.learning_rates = []
        
        self.start_time = time.time()
        self.doc_timer = time.time()
        self.ema_loss = None
        self.alpha = 0.05 
        
    def log_document(self, doc_idx, loss_val, current_lr):
        current_time = time.time()
        elapsed_for_doc = current_time - self.doc_timer
        self.doc_timer = current_time 
        
        docs_s = 1.0 / max(elapsed_for_doc, 1e-5)
        
        if self.ema_loss is None:
            self.ema_loss = loss_val
        else:
            self.ema_loss = self.alpha * loss_val + (1.0 - self.alpha) * self.ema_loss
            
        self.doc_indices.append(doc_idx)
        self.raw_losses.append(loss_val)
        self.smoothed_losses.append(self.ema_loss)
        self.docs_per_sec.append(docs_s)
        self.learning_rates.append(current_lr)

    def generate_dashboard(self, epoch_num):
        if len(self.raw_losses) < 2:
            return
            
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"LLM Training Dashboard - Epoch {epoch_num}", fontsize=16, fontweight='bold')
        
        axs[0, 0].plot(self.doc_indices, self.raw_losses, alpha=0.2, color='royalblue', label='Raw Loss')
        axs[0, 0].plot(self.doc_indices, self.smoothed_losses, color='darkblue', linewidth=2, label='EMA Trend')
        axs[0, 0].set_title("Loss Convergence (Lower is Better)")
        axs[0, 0].set_xlabel("Documents Processed")
        axs[0, 0].set_ylabel("Cross Entropy Loss")
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)
        axs[0, 0].legend()
        
        smoothed_speed = [sum(self.docs_per_sec[max(0, i-20):i+1])/len(self.docs_per_sec[max(0, i-20):i+1]) for i in range(len(self.docs_per_sec))]
        axs[0, 1].plot(self.doc_indices, smoothed_speed, color='forestgreen', alpha=0.8)
        axs[0, 1].set_title("System Throughput (Stability)")
        axs[0, 1].set_xlabel("Documents Processed")
        axs[0, 1].set_ylabel("Documents / Second")
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)
        
        axs[1, 0].plot(self.doc_indices, self.learning_rates, color='darkorange', linewidth=2)
        axs[1, 0].set_title("Learning Rate Evolution")
        axs[1, 0].set_xlabel("Documents Processed")
        axs[1, 0].set_ylabel("Learning Rate")
        axs[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[1, 0].grid(True, linestyle='--', alpha=0.6)
        
        axs[1, 1].axis('off')
        total_elapsed = (time.time() - self.start_time) / 3600.0
        avg_speed = sum(self.docs_per_sec) / len(self.docs_per_sec)
        
        summary_text = (
            f"--- Run Status Summary ---\n\n"
            f"Total Running Time: {total_elapsed:.2f} Hours\n"
            f"Current Document Step: {self.doc_indices[-1]:,}\n"
            f"Starting Training Loss: {self.raw_losses[0]:.4f}\n"
            f"Current Smooth Loss: {self.ema_loss:.4f}\n"
            f"Average Processing Speed: {avg_speed:.2f} docs/sec\n\n"
            f"Dashboard Last Updated: Live"
        )
        axs[1, 1].text(0.1, 0.3, summary_text, fontsize=12, family='monospace',
                      bbox=dict(facecolor='wheat', alpha=0.3, boxstyle='round,pad=1'))
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/epoch_{epoch_num}_progress.png", dpi=150)
        plt.close()

# ==========================================
# 7. UNROLLED TRAINING LOOP WITH CHECKPOINTS
# ==========================================
def main():
    # --- CONFIGURATION ---
    CHUNK_SIZE = 2048           
    ACCUMULATION_STEPS = 16     
    EPOCHS = 4
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.1
    GRAD_CLIP = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- NEWLINE SPAM CONTROL ---
    NEWLINE_TOKEN_ID = 97       # Derived directly from your vocab.json layout
    NEWLINE_PENALTY_COEF = 0.05 # Scales how aggressively to penalize wrongful newlines (try 0.02 - 0.1)

    VOCAB_SIZE = 98
    N_LAYER = 6
    N_HEAD = 6
    D_MODEL = 384
    
    # --- CHECKPOINT SETTINGS ---
    CHECKPOINT_DIR = "checkpoints"
    SAVE_EVERY_N_DOCS = 5000  
    RESUME_FROM_CHECKPOINT = "checkpoints/ckpt_epoch_2_doc_230000.pt" 

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Training securely with infinite context window on: {DEVICE}")

    dataset = DocumentDataset("dataset_tokens.bin")
    dataloader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=True, 
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    model = CustomLargeLanguageModel(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL,
        n_layer=N_LAYER, n_head=N_HEAD
    ).to(DEVICE)

    decay_params = [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2 and p.requires_grad]
    optimizer = optim.AdamW(
        [{"params": decay_params, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=LEARNING_RATE, eps=1e-8,
    )

    total_steps = (EPOCHS * sum([math.ceil(dataset.lengths[i] / CHUNK_SIZE) for i in range(len(dataset))])) // ACCUMULATION_STEPS
    warmup_steps = max(1, total_steps // 20)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda step: step / warmup_steps if step < warmup_steps else max(0.1, 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))))
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # --- RESUME LOGIC ---
    start_epoch = 0
    start_doc = 0
    global_step = 0

    if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
        print(f"\n[!] Loading checkpoint: {RESUME_FROM_CHECKPOINT}")
        checkpoint = torch.load(RESUME_FROM_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        start_doc = checkpoint['doc_idx'] + 1  
        global_step = checkpoint['global_step']
        print(f"[!] Resumed successfully! Starting from Epoch {start_epoch+1}, Document {start_doc}\n")

    tracker = TrainingTracker()

    for epoch in range(start_epoch, EPOCHS):
        for doc_idx, (raw_input_ids, raw_target_ids) in enumerate(dataloader):
            if epoch == start_epoch and doc_idx < start_doc:
                continue

            raw_input_ids = raw_input_ids.view(-1).to(DEVICE)
            raw_target_ids = raw_target_ids.view(-1).to(DEVICE)
            
            total_tokens = raw_input_ids.size(0)
            past_kv = None  
            
            for start_idx in range(0, total_tokens, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_tokens)
                
                input_chunk = raw_input_ids[start_idx:end_idx].unsqueeze(0)
                target_chunk = raw_target_ids[start_idx:end_idx].unsqueeze(0)
                
                if input_chunk.size(1) < 2:
                    break

                with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    logits, loss, current_kv = model(
                        input_chunk, 
                        target_chunk, 
                        past_kv=past_kv, 
                        seq_start_pos=start_idx
                    )
                    
                    # --- LIVE NEWLINE SPAM REGULARIZATION PENALTY ---
                    # Compute probabilities from logits across the chunk sequence
                    probs = F.softmax(logits, dim=-1)
                    newline_probs = probs[..., NEWLINE_TOKEN_ID] # shape: (1, seq_len)
                    
                    # Target mask: everywhere it's NOT padding (-100) and NOT a true newline (97)
                    wrongful_newline_mask = (target_chunk != -100) & (target_chunk != NEWLINE_TOKEN_ID)
                    
                    if wrongful_newline_mask.any():
                        # Isolate the probability mass assigned to newlines where it doesn't belong
                        incorrect_newline_allocations = newline_probs[wrongful_newline_mask]
                        # Penalize the mean probability mass of wrongful newlines
                        newline_penalty = NEWLINE_PENALTY_COEF * incorrect_newline_allocations.mean()
                        loss = loss + newline_penalty

                    # Apply accumulation normalization scaling factor
                    loss = loss / ACCUMULATION_STEPS

                loss.backward()
                
                past_kv = current_kv
                global_step += 1

                if global_step % ACCUMULATION_STEPS == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # --- METRICS LOGGING ---
            if doc_idx % 10 == 0:
                current_loss_val = loss.item() * ACCUMULATION_STEPS
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Doc {doc_idx:04d} | Current Loss: {current_loss_val:.4f}")
                
                tracker.log_document(doc_idx, current_loss_val, current_lr)
                tracker.generate_dashboard(epoch_num=epoch+1)

            # --- MID-EPOCH CHECKPOINTING ---
            if doc_idx > 0 and doc_idx % SAVE_EVERY_N_DOCS == 0:
                ckpt_path = f"{CHECKPOINT_DIR}/ckpt_epoch_{epoch+1}_doc_{doc_idx}.pt"
                torch.save({
                    'epoch': epoch,
                    'doc_idx': doc_idx,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, ckpt_path)
                print(f"  --> Saved mid-epoch backup: {ckpt_path}")

        start_doc = 0

        # --- END OF EPOCH CHECKPOINTING ---
        epoch_ckpt_path = f"{CHECKPOINT_DIR}/ckpt_epoch_{epoch+1}_complete.pt"
        torch.save({
            'epoch': epoch + 1,  
            'doc_idx': 0,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, epoch_ckpt_path)
        print(f"==> Epoch [{epoch+1}/{EPOCHS}] complete. Saved: {epoch_ckpt_path}\n")

    torch.save(model.state_dict(), "final_model_weights.pt")
    print("Training completely finished. Saved final_model_weights.pt")

if __name__ == "__main__":
    main()