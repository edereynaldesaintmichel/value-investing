import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import transformers
print(transformers.__version__)

import os
import glob
import random
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import pickle
import gc

# --- 1. Configuration ---
class Config:
    # Data and Model Paths
    DATA_DIR = "/Users/eloireynal/Documents/My projects/crawl_data/sanitized_txt/"
    BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
    CHECKPOINT_DIR = "checkpoints"
    CACHE_DIR = "embedding_cache"
    
    # Hierarchical Model Architecture
    EMBEDDING_DIM = 768
    NUM_LAYERS = 3
    NUM_ATTENTION_HEADS = 4
    FFN_DIM_MULTIPLIER = 4
    
    # Training Parameters
    EPOCHS = 7
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    TRAIN_SPLIT_RATIO = 0.9
    
    # Data Processing
    MAX_CONTEXT_LEN = 8192
    MIN_CONTEXT_LEN = 512
    MIN_SUB_SEQUENCES = 2
    MAX_SUB_SEQUENCES = 128
    
    # Preprocessing
    MAX_EXAMPLES_TO_GENERATE = 100000
    EXAMPLES_PER_FILE = 5
    PREPROCESSING_BATCH_SIZE = 8
    
    # Token ratio calculation
    SAMPLE_FILES_FOR_RATIO = 20  # Number of files to sample for ratio calculation
    
    # System
    DEVICE = "mps"

# --- Setup Logging and Directories ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)

# --- 2. Token Ratio Calculation ---
def calculate_token_char_ratio(file_paths, tokenizer, num_samples=20):
    """
    Calculate the average token-to-character ratio from sample files.
    """
    sample_files = random.sample(file_paths, min(num_samples, len(file_paths)))
    
    total_chars = 0
    total_tokens = 0
    
    logging.info(f"Calculating token-to-character ratio from {len(sample_files)} sample files...")
    
    for file_path in tqdm(sample_files, desc="Analyzing token ratio"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Take multiple chunks from each file for better average
            chunk_size = 10000  # 10k characters per chunk
            for i in range(0, min(len(text), 50000), chunk_size):  # Max 5 chunks per file
                chunk = text[i:i+chunk_size]
                if len(chunk) < 100:
                    continue
                    
                tokenized = tokenizer(chunk, return_tensors="pt", truncation=False)
                num_tokens = tokenized['input_ids'].shape[1]
                
                total_chars += len(chunk)
                total_tokens += num_tokens
                
        except Exception as e:
            logging.warning(f"Error processing {file_path} for ratio calculation: {e}")
            continue
    
    if total_chars == 0:
        logging.warning("No valid text found for ratio calculation. Using default ratio.")
        return 0.3  # Fallback ratio (roughly 3.3 chars per token)
    
    ratio = total_tokens / total_chars
    chars_per_token = total_chars / total_tokens
    
    logging.info(f"Token-to-character ratio: {ratio:.4f} tokens per char")
    logging.info(f"Character-to-token ratio: {chars_per_token:.2f} chars per token")
    
    return ratio

# --- 3. Pre-processing and Caching Functions ---
def split_text(text, number_of_chunks):
    text_length = len(text)
    chunk_length = text_length / number_of_chunks
    chunks = []
    split_regex = r'(?=\.\.\.|[\.?!;:—–] |\r\n|\r|\n)'
    splitted_text = re.split(split_regex, text)
    if len(splitted_text) < number_of_chunks:
        return splitted_text
    
    cumulative_length = []
    length = 0
    for token in splitted_text:
        length += len(token)
        cumulative_length.append(length)
    
    current_running_length = 0
    prev_length = 0
    last_chunk_index = 0
    for i, length in enumerate(cumulative_length):
        previous_running_length = current_running_length
        current_running_length += length - prev_length
        if current_running_length > chunk_length:
            dist1 = current_running_length - chunk_length
            dist2 = chunk_length - previous_running_length
            if dist2 < dist1:
                chunks.append(text[prev_length - previous_running_length:prev_length])
                current_running_length = length - prev_length
            else:
                chunks.append(text[length - current_running_length:length])
                current_running_length = 0
        prev_length = length
    # loss = sum([(len(chunk) - chunk_length)**2 for chunk in chunks])**.5
    return chunks

def preprocess_and_cache_embeddings(file_paths, tokenizer, token_char_ratio, cache_file_prefix, 
                                    max_examples=None, examples_per_file=4):
    """
    Pre-process files and create cached embeddings using character-based chunking
    with the calculated token-to-character ratio.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_embeddings.pkl")
    
    if os.path.exists(cache_file):
        logging.info(f"Cache file {cache_file} already exists. Loading...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logging.info(f"Creating new cache file: {cache_file}")
    
    # Convert token lengths to character lengths using the ratio
    max_chars = int(Config.MAX_CONTEXT_LEN / token_char_ratio)
    min_chars = int(Config.MIN_CONTEXT_LEN / token_char_ratio)
    
    logging.info(f"Using character limits: {min_chars} to {max_chars} (based on token limits {Config.MIN_CONTEXT_LEN} to {Config.MAX_CONTEXT_LEN})")
    
    texts_to_process = []
    metadata = []
    
    pbar = tqdm(file_paths, desc=f"Reading files for {cache_file_prefix}")
    for file_path in pbar:
        if max_examples and len(texts_to_process) >= max_examples:
            break
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
            
            if len(full_text.strip()) < min_chars:
                continue
            
            for _ in range(examples_per_file):
                if max_examples and len(texts_to_process) >= max_examples:
                    break
                
                # Calculate desired chunk size in characters
                desired_chars = random.randint(min_chars, max_chars)
                
                # If text is longer than desired, randomly sample a starting point
                if len(full_text) > desired_chars:
                    start_idx = random.randint(0, len(full_text) - desired_chars)
                    chunk_text = full_text[start_idx:start_idx + desired_chars]
                else:
                    chunk_text = full_text
                
                # Find a good ending point (sentence boundary if possible)
                if len(chunk_text) > 100:
                    # Try to end at a sentence boundary
                    last_period = chunk_text.rfind('.')
                    last_exclaim = chunk_text.rfind('!')
                    last_question = chunk_text.rfind('?')
                    last_newline = chunk_text.rfind('\n')
                    
                    last_boundary = max(last_period, last_exclaim, last_question, last_newline)
                    if last_boundary > len(chunk_text) * 0.8:  # If boundary is in last 20% of text
                        chunk_text = chunk_text[:last_boundary + 1]
                
                # Estimate number of tokens for logging/validation
                estimated_tokens = int(len(chunk_text) * token_char_ratio)
                
                # Split into sub-sequences
                num_sub_sequences = random.randint(
                    Config.MIN_SUB_SEQUENCES, 
                    min(Config.MAX_SUB_SEQUENCES, len(chunk_text) // 200)  # Rough estimate: 200 chars per meaningful chunk
                )
                sub_sequence_texts = split_text(chunk_text, num_sub_sequences)
                
                if len(sub_sequence_texts) < 2:
                    continue
                
                texts_to_process.append((chunk_text, sub_sequence_texts))
                metadata.append({
                    'source_file': file_path,
                    'num_sequences': len(sub_sequence_texts),
                    'chunk_chars': len(chunk_text),
                    'estimated_tokens': estimated_tokens
                })
                
        except Exception as e:
            logging.warning(f"Error processing {file_path}: {e}")
            continue
    
    # Save the preprocessed data
    torch.save(texts_to_process, os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_texts_to_process.pt"))
    torch.save(metadata, os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_metadata.pt"))
    
    # Log statistics
    if metadata:
        avg_chars = np.mean([m['chunk_chars'] for m in metadata])
        avg_tokens = np.mean([m['estimated_tokens'] for m in metadata])
        avg_sequences = np.mean([m['num_sequences'] for m in metadata])
        
        logging.info(f"Preprocessing complete:")
        logging.info(f"  - Total examples: {len(texts_to_process)}")
        logging.info(f"  - Avg chunk size: {avg_chars:.0f} chars (~{avg_tokens:.0f} tokens)")
        logging.info(f"  - Avg sub-sequences per chunk: {avg_sequences:.1f}")
    
    return texts_to_process

# --- 4. Main Training Script ---
def main():
    logging.info(f"Using device: {Config.DEVICE}")
    
    # --- Load Tokenizer ---
    logging.info(f"Loading tokenizer: {Config.BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)
    
    # --- Load file paths ---
    logging.info(f"Loading data from: {Config.DATA_DIR}")
    all_files = glob.glob(os.path.join(Config.DATA_DIR, "*.md"))
    
    if not all_files:
        logging.error(f"No .md files found in {Config.DATA_DIR}")
        return
    
    logging.info(f"Found {len(all_files)} files")
    
    # --- Calculate token-to-character ratio ---
    token_char_ratio = calculate_token_char_ratio(all_files, tokenizer, Config.SAMPLE_FILES_FOR_RATIO)
    
    # --- Preprocess and cache ---
    train_examples = preprocess_and_cache_embeddings(
        all_files, 
        tokenizer,
        token_char_ratio,
        "website", 
        max_examples=Config.MAX_EXAMPLES_TO_GENERATE, 
        examples_per_file=Config.EXAMPLES_PER_FILE
    )
    
    logging.info("Preprocessing complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if torch.cuda.is_available():
            logging.info("Script finished or crashed. Clearing CUDA cache to release GPU memory.")
            torch.cuda.empty_cache()
            gc.collect()