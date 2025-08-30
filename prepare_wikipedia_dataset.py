import os
import re
import random
import pickle
import logging
import torch
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict, Any
import gzip
import json
import requests
from datasets import load_dataset
from helpers.chunk_texts_intelligently import split_text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    MAX_CONTEXT_LEN = 8000
    MIN_CONTEXT_LEN = 512
    MAX_SUB_SEQUENCE_LENGTH = 512
    MIN_SUB_SEQUENCE_LENGTH = 10
    
    # Preprocessing
    MAX_EXAMPLES_TO_GENERATE = 100000
    PREPROCESSING_BATCH_SIZE = 8
    

def ensure_cache_dir():
    """Ensure cache directory exists"""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

def download_wikipedia():
    wiki_dataset = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en",  # English Wikipedia snapshot
        split="train",
        streaming=True  # Use streaming to avoid loading entire dataset into memory
    )
    logging.info("Successfully loaded wikimedia/wikipedia dataset")
    return wiki_dataset, "wikimedia"


def extract_text_from_dataset_item(item: Dict, dataset_type: str) -> Tuple[str, str]:
    """
    Extract text and title from different dataset formats.
    
    Returns:
        Tuple of (text, title)
    """
        # wikimedia/wikipedia format
    text = item.get('text', '')
    title = item.get('title', 'Unknown')
    
    return text, title

def process_wikipedia_to_training_examples(
    num_examples: int = 5000,
    cache_file_prefix: str = "wikipedia"
) -> Tuple[List[Tuple[str, List[str]]], List[Dict[str, Any]]]:
    """
    Process Wikipedia articles into training examples with the same structure
    as the original function.
    
    Args:
        num_examples: Total number of training examples to create
        examples_per_article: Number of examples to extract from each article
        cache_file_prefix: Prefix for cache files
        
    Returns:
        Tuple of (texts_to_process, metadata) with same structure as original
    """
    ensure_cache_dir()
    
    # Check if processed data already exists
    texts_cache_file = os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_texts_to_process.pkl")
    metadata_cache_file = os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_metadata.pkl")
    
    if os.path.exists(texts_cache_file) and os.path.exists(metadata_cache_file):
        logging.info("Loading cached processed data...")
        with open(texts_cache_file, 'rb') as f:
            texts_to_process = pickle.load(f)
        with open(metadata_cache_file, 'rb') as f:
            metadata = pickle.load(f)
        logging.info(f"Loaded {len(texts_to_process)} cached examples")
        return texts_to_process, metadata
    
    # Download Wikipedia or alternative dataset
    dataset, dataset_type = download_wikipedia()
    
    texts_to_process = []
    metadata = []
    articles_processed = 0
    
    # Create progress bar
    pbar = tqdm(total=num_examples, desc="Creating training examples")
    
    # Handle both streaming and non-streaming datasets
    if hasattr(dataset, '__iter__'):
        dataset_iter = iter(dataset)
    else:
        dataset_iter = dataset
    
    max_target_chars = int(Config.MAX_CONTEXT_LEN * 4)
    min_target_chars = int(Config.MIN_CONTEXT_LEN * 4)
    min_sub_chars = int(Config.MIN_SUB_SEQUENCE_LENGTH * 4)
    max_sub_chars = int(Config.MAX_SUB_SEQUENCE_LENGTH * 4)
    
    # Process articles from the dataset
    for article in dataset_iter:
        if len(texts_to_process) >= num_examples:
            break
        
        full_text, article_title = extract_text_from_dataset_item(article, dataset_type)
        
        if len(full_text.strip()) < Config.MIN_CONTEXT_LEN * 4 * 2:
            continue
        
        target_chunk_size = random.randint(min_target_chars, max_target_chars)
        target_chunks = split_text(full_text, chunk_length=target_chunk_size)
        subtext_chunk_size = random.randint(min_sub_chars, min(max_sub_chars, target_chunk_size//3)) # I want at least 3 sub_sequences
        
        for target_chunk_text in target_chunks:
            estimated_tokens = int(len(target_chunk_text) / 4)
            sub_sequence_texts = split_text(target_chunk_text, subtext_chunk_size)
            texts_to_process.append((target_chunk_text, sub_sequence_texts))
            metadata.append({
                'source_file': f"{dataset_type}:{article_title}",
                'num_sequences': len(sub_sequence_texts),
                'chunk_chars': len(target_chunk_text),
                'estimated_tokens': estimated_tokens
            })
        
        articles_processed += 1
        pbar.update(1)
        
        # Log progress every 100 articles
        # if articles_processed % 100 == 0:
        #     logging.info(f"Processed {articles_processed} articles, created {len(texts_to_process)} examples")
                
    
    pbar.close()
    
    logging.info(f"Created {len(texts_to_process)} training examples from {articles_processed} articles")
    logging.info(f"Dataset type used: {dataset_type}")
    
    # Save processed data to cache
    logging.info("Saving processed data to cache...")
    with open(texts_cache_file, 'wb') as f:
        pickle.dump(texts_to_process, f)
    with open(metadata_cache_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # # Also save in PyTorch format for compatibility
    # torch.save(texts_to_process, os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_texts_to_process.pt"))
    # torch.save(metadata, os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_metadata.pt"))
    
    return texts_to_process, metadata

def analyze_dataset(texts_to_process: List[Tuple[str, List[str]]], metadata: List[Dict[str, Any]]):
    """Analyze and print statistics about the created dataset"""
    logging.info("\n" + "="*50)
    logging.info("Dataset Statistics:")
    logging.info("="*50)
    
    total_examples = len(texts_to_process)
    logging.info(f"Total training examples: {total_examples}")
    
    # Check dataset types
    if metadata and 'dataset_type' in metadata[0]:
        dataset_types = set(m.get('dataset_type', 'unknown') for m in metadata)
        logging.info(f"Dataset types used: {', '.join(dataset_types)}")
    
    # Analyze chunk lengths
    chunk_lengths = [len(text[0]) for text in texts_to_process]
    avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_chunk_length = min(chunk_lengths) if chunk_lengths else 0
    max_chunk_length = max(chunk_lengths) if chunk_lengths else 0
    
    logging.info(f"Chunk length - Avg: {avg_chunk_length:.0f}, Min: {min_chunk_length}, Max: {max_chunk_length}")
    
    # Analyze sub-sequences
    num_subsequences = [len(text[1]) for text in texts_to_process]
    avg_subsequences = sum(num_subsequences) / len(num_subsequences) if num_subsequences else 0
    min_subsequences = min(num_subsequences) if num_subsequences else 0
    max_subsequences = max(num_subsequences) if num_subsequences else 0
    
    logging.info(f"Sub-sequences per example - Avg: {avg_subsequences:.1f}, Min: {min_subsequences}, Max: {max_subsequences}")
    
    # Show sample
    if texts_to_process:
        logging.info("\n" + "="*50)
        logging.info("Sample Example:")
        logging.info("="*50)
        sample_idx = random.randint(0, len(texts_to_process) - 1)
        sample_text, sample_subsequences = texts_to_process[sample_idx]
        sample_meta = metadata[sample_idx]
        
        logging.info(f"Source: {sample_meta['source_file']}")
        logging.info(f"Number of sub-sequences: {sample_meta['num_sequences']}")
        logging.info(f"Main chunk preview (first 200 chars): {sample_text[:200]}...")
        logging.info(f"First sub-sequence preview: {sample_subsequences[0][:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia into training examples")
    parser.add_argument('--num-examples', type=int, default=500000, 
                       help='Number of training examples to create')
    parser.add_argument('--examples-per-article', type=int, default=3,
                       help='Number of examples to extract from each article')
    parser.add_argument('--cache-prefix', type=str, default='wikipedia',
                       help='Prefix for cache files')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing cached data')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load existing data and analyze
        texts_cache_file = os.path.join(Config.CACHE_DIR, f"{args.cache_prefix}_texts_to_process.pkl")
        metadata_cache_file = os.path.join(Config.CACHE_DIR, f"{args.cache_prefix}_metadata.pkl")
        
        if os.path.exists(texts_cache_file) and os.path.exists(metadata_cache_file):
            with open(texts_cache_file, 'rb') as f:
                texts_to_process = pickle.load(f)
            with open(metadata_cache_file, 'rb') as f:
                metadata = pickle.load(f)
            analyze_dataset(texts_to_process, metadata)
        else:
            logging.error("No cached data found. Run without --analyze-only first.")
    else:
        # Process Wikipedia and create training examples
        texts_to_process, metadata = process_wikipedia_to_training_examples(
            num_examples=args.num_examples,
            cache_file_prefix=args.cache_prefix
        )
        
        # Analyze the created dataset
        analyze_dataset(texts_to_process, metadata)
        
        logging.info(f"\nData saved to {Config.CACHE_DIR}/")
        logging.info("Files created:")
        logging.info(f"  - {args.cache_prefix}_texts_to_process.pkl")
        logging.info(f"  - {args.cache_prefix}_texts_to_process.pt")
        logging.info(f"  - {args.cache_prefix}_metadata.pkl")
        logging.info(f"  - {args.cache_prefix}_metadata.pt")

if __name__ == "__main__":
    main()