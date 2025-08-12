import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

import os
import glob
import random
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# --- 1. Configuration ---
class Config:
    # Data and Model Paths
    DATA_DIR = "/Users/eloireynal/Documents/My projects/crawl_data/txt/" # <<< IMPORTANT: SET THIS PATH
    BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
    CHECKPOINT_DIR = "./checkpoints"
    
    # Hierarchical Model Architecture
    EMBEDDING_DIM = 768  # From modern-bert-base
    NUM_LAYERS = 3
    NUM_ATTENTION_HEADS = 4
    FFN_DIM_MULTIPLIER = 4
    
    # Training Parameters
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1 # Start with 1 due to 16GB RAM constraint on M4. Increase if possible.
    TRAIN_SPLIT_RATIO = 0.9
    
    # Data Processing
    MAX_CONTEXT_LEN = 8192 # ModernBERT's context window
    MIN_SUB_SEQUENCES = 2
    MAX_SUB_SEQUENCES = 128 # As requested
    
    # System
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Setup Logging and Directories ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)


# --- 2. Rotary Positional Embedding (RoPE) ---
class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE).
    This is injected into the attention mechanism.
    """
    def __init__(self, dim, base=5000):
        super().__init__()
        self.dim = dim
        self.base = base
        # Precompute theta values for frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _cache_cos_sin(self, seq_len, device):
        if self.seq_len_cached != seq_len:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
    
    def _apply_rotary_emb(self, x, cos, sin):
        # x shape: (batch, n_heads, seq_len, head_dim)
        x_rotated = torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
        return (x * cos) + (x_rotated * sin)

    def forward(self, q, k):
        # q, k shape: (batch, n_heads, seq_len, head_dim)
        seq_len = q.shape[-2]
        device = q.device
        
        self._cache_cos_sin(seq_len, device)
        
        # Apply rotation to queries and keys
        q_rotated = self._apply_rotary_emb(q, self.cos_cached[:seq_len], self.sin_cached[:seq_len])
        k_rotated = self._apply_rotary_emb(k, self.cos_cached[:seq_len], self.sin_cached[:seq_len])
        
        return q_rotated, k_rotated

# --- 3. Hierarchical Model Architecture ---
class HierarchicalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            # Mask needs to be broadcastable to (batch_size, num_heads, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))

class HierarchicalEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.attention = HierarchicalAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        # Pre-LayerNorm
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x

class HierarchicalBert(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ffn_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            HierarchicalEncoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# --- 4. Data Handling ---
def sensible_split(text, num_chunks):
    """Splits text into a number of chunks at sensible points (punctuation/newlines)."""
    # Find all potential split points
    split_points = [m.start() for m in re.finditer(r'[.?!]\s|\n', text)]
    
    if len(split_points) < num_chunks - 1:
        # Not enough sensible points, fall back to rough splitting
        chunk_size = len(text) // num_chunks
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)][:num_chunks]

    # Randomly select split points
    chosen_indices = sorted(random.sample(split_points, num_chunks - 1))
    
    chunks = []
    start_idx = 0
    for split_idx in chosen_indices:
        chunks.append(text[start_idx:split_idx+1].strip())
        start_idx = split_idx + 1
    chunks.append(text[start_idx:].strip())
    
    return [chunk for chunk in chunks if chunk] # Remove empty strings

class TextFileDataset(Dataset):
    def __init__(self, file_paths, tokenizer):
        self.file_paths = file_paths
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()

        # 1. Chop down to a single chunk that fits ModernBERT's context
        tokenized_full = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=Config.MAX_CONTEXT_LEN)
        chunk_text = self.tokenizer.decode(tokenized_full['input_ids'][0], skip_special_tokens=True)

        # 2. Chop chunk into a random number of random-sized sequences
        num_sub_sequences = random.randint(Config.MIN_SUB_SEQUENCES, Config.MAX_SUB_SEQUENCES)
        sub_sequence_texts = sensible_split(chunk_text, num_sub_sequences)
        
        # This can happen with very short texts, ensure we have at least 2 sequences
        if len(sub_sequence_texts) < 2 and len(chunk_text) > 10:
             mid = len(chunk_text) // 2
             sub_sequence_texts = [chunk_text[:mid], chunk_text[mid:]]
        elif len(sub_sequence_texts) < 2:
             # Handle very short texts by duplicating
             sub_sequence_texts = [chunk_text, chunk_text]

        return {"id": file_path, "whole_chunk": chunk_text, "sub_sequences": sub_sequence_texts}

def create_collate_fn(tokenizer, base_model, device):
    def collate_fn(batch):
        # This function runs on the main process
        # It takes a list of dictionaries from the dataset
        
        whole_chunks_text = [item['whole_chunk'] for item in batch]
        sub_sequences_list = [item['sub_sequences'] for item in batch]
        
        # We must not compute gradients for the base model
        with torch.no_grad():
            # Get target embeddings ([CLS] of the whole chunk)
            target_tokens = tokenizer(whole_chunks_text, return_tensors='pt', padding=True, truncation=True, max_length=Config.MAX_CONTEXT_LEN).to(device)
            target_outputs = base_model(**target_tokens)
            target_cls_embeddings = target_outputs.last_hidden_state[:, 0, :] # (batch_size, embed_dim)

            input_embeddings_list = []
            original_lengths = []
            
            # Get input embeddings ([CLS] of each sub-sequence)
            for sub_sequences in sub_sequences_list:
                original_lengths.append(len(sub_sequences))
                sub_tokens = tokenizer(sub_sequences, return_tensors='pt', padding=True, truncation=True, max_length=Config.MAX_CONTEXT_LEN).to(device)
                sub_outputs = base_model(**sub_tokens)
                cls_embeddings = sub_outputs.last_hidden_state[:, 0, :]
                input_embeddings_list.append(cls_embeddings)
        
        # Pad the input sequences to the max length in the batch
        padded_input_embeddings = nn.utils.rnn.pad_sequence(input_embeddings_list, batch_first=True, padding_value=0.0)
        
        # Create attention mask for the hierarchical model
        max_len = padded_input_embeddings.size(1)
        attention_mask = torch.arange(max_len)[None, :].to(device) < torch.tensor(original_lengths)[:, None].to(device)
        # Reshape mask for multi-head attention: (batch_size, 1, 1, seq_len)
        attention_mask = attention_mask[:, None, None, :]

        return {
            "input_embeddings": padded_input_embeddings.to(device),
            "attention_mask": attention_mask.to(device),
            "target_embedding": target_cls_embeddings.to(device),
            "original_lengths": torch.tensor(original_lengths).to(device)
        }
    return collate_fn


# --- 5. Training Script ---
def main():
    logging.info(f"Using device: {Config.DEVICE}")
    
    # --- Load Base Model and Tokenizer ---
    logging.info(f"Loading base model: {Config.BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)
    base_model = AutoModel.from_pretrained(Config.BASE_MODEL_NAME).to(Config.DEVICE)
    base_model.eval() # Set to evaluation mode
    for param in base_model.parameters():
        param.requires_grad = False # FREEZE a.k.a. NO_GRAD HERE
        
    # --- Load Data ---
    logging.info(f"Loading data from: {Config.DATA_DIR}")
    all_files = glob.glob(os.path.join(Config.DATA_DIR, "*.txt"))
    if not all_files:
        logging.error(f"No .txt files found in {Config.DATA_DIR}. Please check the path.")
        return
        
    train_files, val_files = train_test_split(all_files, train_size=Config.TRAIN_SPLIT_RATIO, random_state=42)
    logging.info(f"Found {len(all_files)} files. Training on {len(train_files)}, validating on {len(val_files)}.")

    train_dataset = TextFileDataset(train_files, tokenizer)
    val_dataset = TextFileDataset(val_files, tokenizer)
    
    # Custom collate function to handle dynamic data processing
    dynamic_collate_fn = create_collate_fn(tokenizer, base_model, Config.DEVICE)
    
    # Note: num_workers=0 is often safer on macOS
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=dynamic_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn, num_workers=0)
    
    # --- Initialize Hierarchical Model, Loss, Optimizer ---
    logging.info("Initializing HierarchicalBert model")
    hierarchical_model = HierarchicalBert(
        embed_dim=Config.EMBEDDING_DIM,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_ATTENTION_HEADS,
        ffn_dim=Config.EMBEDDING_DIM * Config.FFN_DIM_MULTIPLIER
    ).to(Config.DEVICE)
    print("Number of params: " + str(sum(p.numel() for p in hierarchical_model.parameters())/1e6) + "M")
    optimizer = torch.optim.AdamW(hierarchical_model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # --- Training Loop ---
    for epoch in range(Config.EPOCHS):
        # Training phase
        hierarchical_model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            
            output_embeddings = hierarchical_model(batch['input_embeddings'], batch['attention_mask'])
            
            # Get the embedding of the *last actual* sequence for each item in the batch
            last_seq_indices = batch['original_lengths'] - 1
            last_token_embeddings = output_embeddings[torch.arange(output_embeddings.size(0)), last_seq_indices]

            loss = criterion(last_token_embeddings, batch['target_embedding'])
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        hierarchical_model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]")

        with torch.no_grad():
            for batch in val_pbar:
                output_embeddings = hierarchical_model(batch['input_embeddings'], batch['attention_mask'])
                
                last_seq_indices = batch['original_lengths'] - 1
                last_token_embeddings = output_embeddings[torch.arange(output_embeddings.size(0)), last_seq_indices]

                loss = criterion(last_token_embeddings, batch['target_embedding'])
                total_val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})

        avg_val_loss = total_val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(hierarchical_model.state_dict(), checkpoint_path)
            logging.info(f"Validation loss decreased. Saving model to {checkpoint_path}")

    logging.info("Training complete.")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()