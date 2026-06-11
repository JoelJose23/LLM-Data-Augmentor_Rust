import requests
import json
from datasets import load_dataset

print("Loading Datasets....")
# Standard message array datasets
agent_dataset = load_dataset("arcee-ai/agent-data", split="train")
chat_dataset = load_dataset("LMSYS/lmsys-chat-1m", split="train")

# Columnar datasets (Function calling & Instruction tuning)
xlam_dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
orca_dataset = load_dataset("Open-Orca/OpenOrca", split="train")

Rust_URL = "http://127.0.0.1:3000/process_text"

def parse_messages_to_chatml(messages):
    """Converts standard message lists/ShareGPT formats into ChatML blocks."""
    text_block = ""
    for msg in messages:
        role = msg.get('role') or msg.get('from')
        content = msg.get('content') or msg.get('value')
        
        if role in ['human', 'user']:
            role = 'user'
        elif role in ['gpt', 'assistant']:
            role = 'assistant'
        elif role == 'system':
            # Skip alternative system prompts to protect the core model persona
            continue
            
        text_block += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text_block

def parse_xlam_to_chatml(row):
    """Converts flat XLAM rows into un-indented, structured ChatML blocks."""
    def canonicalize_field(field_data):
        if isinstance(field_data, str):
            try:
                parsed = json.loads(field_data)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                return field_data.strip()
        elif isinstance(field_data, (list, dict)):
            return json.dumps(field_data, ensure_ascii=False)
        return str(field_data).strip()

    tools_payload = canonicalize_field(row.get("tools", []))
    answers_payload = canonicalize_field(row.get("answers", []))
    user_query = str(row.get("query", "")).strip()

    text_block = (
        f"<|im_start|>system\nTools: {tools_payload}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n{answers_payload}<|im_end|>\n"
    )
    return text_block

def parse_openorca_to_chatml(row):
    """Converts OpenOrca columns into lean ChatML tokens, filtering prompt pollution."""
    user_query = str(row.get("question", "")).strip()
    assistant_response = str(row.get("response", "")).strip()

    # The 'system_prompt' column is skipped entirely here to keep data footprint clean
    text_block = (
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_response}<|im_end|>\n"
    )
    return text_block

def stream_agent_data():
    print("Streaming Agent and Tool Use trajectories to Rust...")
    for row in agent_dataset:
        messages = row.get('messages') or row.get('conversations')
        if not messages:
            continue
            
        text_block = parse_messages_to_chatml(messages)
        payload = {"text_content": text_block}
        try:
            requests.post(Rust_URL, json=payload)
        except requests.exceptions.ConnectionError:
            print("Error: Is your Rust server running?")
            break

def stream_xlam_data():
    print("Streaming Salesforce XLAM Parallel Function Calling data...")
    for row in xlam_dataset:
        text_block = parse_xlam_to_chatml(row)
        payload = {"text_content": text_block}
        try:
            requests.post(Rust_URL, json=payload)
        except requests.exceptions.ConnectionError:
            print("Error: Connection lost while streaming XLAM tokens.")
            break

def stream_openorca_data():
    print("Streaming Open-Orca Sub-sampled Reasoning Dataset...")
    for idx, row in enumerate(orca_dataset):
        text_block = parse_openorca_to_chatml(row)
        payload = {"text_content": text_block}
        try:
            requests.post(Rust_URL, json=payload)
        except requests.exceptions.ConnectionError:
            print("Error: Connection lost while streaming OpenOrca tokens.")
            break

        # Maintain a balanced data mixture (OpenOrca is massive)
        if idx >= 20000:
            break

def stream_chat_data():
    print("Streaming LMSYS Arena Conversational Dialogue...")
    for idx, row in enumerate(chat_dataset):
        messages = row.get('messages') or row.get('conversations')
        if not messages:
            continue

        text_block = parse_messages_to_chatml(messages)
        payload = {"text_content": text_block}
        try:
            requests.post(Rust_URL, json=payload)
        except requests.exceptions.ConnectionError:
            print("Error: Connection lost while streaming chat tokens.")
            break

        if idx >= 20000:
            break

if __name__ == "__main__":
    stream_agent_data()
    stream_xlam_data()
    stream_openorca_data()
    stream_chat_data()
    print("All datasets successfully piped to Rust")