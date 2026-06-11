# 🧠 Custom Character-Level LLM Engine & Agent

A high-performance, native **Character-Level Large Language Model** architecture built completely from scratch in PyTorch, utilizing a blazingly fast native **Rust (Axum) Tokenizer Server**. 

This repository implements an independent, low-parameter transformer model trained locally on consumer hardware over a massive, carefully curated dataset mixture of **565,875 documents**. It successfully scales from initial architectural entropy collapse (infinite newline loops) to a functional, tool-calling conversational agent capable of complex, multi-clause syntax and semantic reasoning.

---

## 🏗️ Technical Architecture & Configurations

The model intentionally leverages a compact, high-density matrix configuration designed to fit comfortably within consumer GPU high-speed cache layers (L1/L2) and VRAM pools, yielding exceptionally fast, near-zero overhead inference.

```python
# Core Hyperparameters (Must match training and inference pipelines)
VOCAB_SIZE = 98      # Dense character-level map
D_MODEL = 384        # Hidden embedding dimension
N_LAYER = 6          # Transformer blocks
N_HEAD = 6           # Multi-head attention heads
HEAD_SIZE = 64       # d_model // n_head
MAX_SEQ_LEN = 4096   # Dynamic sequence cache length
```

### Key Architectural Pillars
* **Rotary Positional Embeddings (RoPE):** Offset-aware relative positional encoding applied directly to query and key projections per block, avoiding absolute token lookups and maximizing long-range contextual awareness.
* **Causal Flash-Style Attention:** Fully integrated with PyTorch's `scaled_dot_product_attention` for fast, mathematically optimized scaling, operating alongside an autoregressive Key-Value (KV) cache tracking framework.
* **Tied Embeddings:** Weights of the token embedding matrix are strictly mirrored onto the final classification head (`lm_head`), reducing overall parameter footprint while preserving semantic alignment.

---

## 🗂️ Dataset Curriculum Mixture

The training protocol processes a vast, elite-tier alignment dataset containing **565,875 high-quality records** piped sequentially. The curriculum balances human conversation fluidities, agentic step trajectories, and rigid functional reasoning:

1. **`arcee-ai/agent-data`**: Deep multi-step instruction compliance and multi-turn trajectory execution.
2. **`LMSYS/lmsys-chat-1m`**: High-entropy, organic human-to-assistant conversational dialogue.
3. **`Open-Orca/OpenOrca`**: Premium reasoning, thought chains, and instruction-tuning logic blocks.
4. **`Salesforce/xlam-function-calling-60k`**: Parallel tool definitions and structured programmatic text outputs.

### Tokenization Strategy
All data streams are transformed into clean **ChatML sequences** and piped live to the Rust server. Rather than utilizing sub-word Byte-Pair Encoding (BPE), the vocabulary uses a precise **98-token character-level mapping** (`vocab.json`). Structural blocks are evaluated character-by-character:
```text
<|im_start|>system
Tools: {json_payload}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant

```

---

## 🦀 Native Rust Tokenizer Server (`main.rs`)

To eliminate Python's string-parsing bottlenecks, a native, multi-threaded tokenization engine was implemented in Rust using the **Axum** web framework. 

* **Dual-Pass Binary Packer:** In pass one, the server indexes JSONL structural bounds. In pass two, it stream-parses records directly into a performance-tuned binary file (`dataset_tokens.bin`) using explicit little-endian byte layouts (`<I`).
* **Live Inference API**: Exposes highly optimized web endpoints for sub-millisecond execution during active interaction:
  * `POST /encode`: Map raw string inputs directly down to integer index vectors.
  * `POST /decode`: Instantly reverse token streams back to screen-printable strings.

---

## 📈 Training Checkpoint Progression (Epoch 1)

Training an LLM from scratch is a lesson in patience. Below is the empirical evolution of the model's outputs as it navigated architectural entropy and converged into structured speech.

### 🚨 Milestone 1: The Newline Entropy Loop (Docs 0 – 100k)

Early in training, the model discovers that the newline token (`\n`) is universally frequent across all datasets. To cheat Cross-Entropy loss without understanding language syntax yet, it collapses into an endless repetition loop.

* **The Symptom:** Model instantly floods the console with infinite blank lines.
* **The Fix:** Continued natural training paired with a dynamic **wrongful newline probability penalty** integrated directly into the backpropagation loop to suppress formatting overconfidence.

```text
======================================================================
PROMPT: Hello, how are you?
======================================================================
RESPONSE: 
\n
\n
\n
\n
\n
======================================================================
```

<br>

### 🛠️ Milestone 2: The Proto-Agent Bracket Glitch (Docs 100k – 400k)

The model successfully intercepts complex system instructions and handles heavy structural tool-routing logic, but displays mechanical spelling artifacts and character-adjacent hallucinations due to stabilizing attention heads.

* **The Symptom:** Errant JSON brackets, escaped backslashes, and unclosed tracking tags (`<|im_end|`).

```text
======================================================================
PROMPT: Hello, how is the weather today?
======================================================================
RESPONSE: 
<tool_call>}'tool_name': 'get_weather_api', 'tool_arguments': }'q': 'Hello, how is the weather today?'\</tool_call><|im_end|
======================================================================

======================================================================
PROMPT: How are you feeling today?
======================================================================
RESPONSE: 
What is your feeling when you're feeling? Are you proud of your feelings, and I'm here to help you become more aware of your thoughts, feelings, and experiences?<|im_end|>
======================================================================
```

---

## 🚀 Running Local Inference

To spin up active validation on local consumer hardware:

1. **Launch the Tokenizer Engine:**
   ```bash
   # Run the server to bind onto http://127.0.0.1:3000
   cargo run --release
   ```
2. **Execute Interactive Generation:**
   ```bash
   # Automatically maps checkpoints onto CUDA and initializes auto-regressive KV-cache streams
   python LLM_For_Interaction/inference.py
   ```

---

## 📝 Lessons Learned for Future Models (V2 Blueprint)
* **Atomic Special Tokens:** In a character-level setup, structural boundaries like `<|im_start|>` require the model to predict 10 tokens flawlessly. In V2, these will be injected directly as discrete, atomic integers (IDs 98 and 99) within the `vocab.json` matrix and Rust string match loops to bypass sequential parsing degradation entirely.
* **Decay Flooring:** Rather than allowing the Cosine Learning Rate Scheduler to decay all the way to `10%`, flattening the minimum floor to a stable mid-range value (e.g., `30%` or `1e-4`) maintains steady, consistent momentum across multiple epochs without stalling.
