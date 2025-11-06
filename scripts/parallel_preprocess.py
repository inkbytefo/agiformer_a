# scripts/parallel_preprocess.py

# Developer: inkbytefo
# Modified: 2025-11-06

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import os

# Projenin kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agiformer.language.morpho_splitter import RegexSplitter

# Global RegexSplitter nesnesi (her bir alt işlemde bir kez başlatılacak)
splitter = None

def init_worker():
    """Her bir alt işlem (worker) için RegexSplitter'ı başlatır."""
    global splitter
    print(f"Initializing RegexSplitter for process {os.getpid()}...")
    splitter = RegexSplitter()

def process_chunk(lines: list[str]) -> list[str]:
    """
    Bir grup satırı (chunk) morfolojik olarak işler.
    Bu fonksiyon her bir alt işlemde çalışır.
    """
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            analysis = splitter.split_sentence(line)
            morphemes = []
            for word_analysis in analysis["kelimeler"]:
                root = word_analysis["kök"]
                if root:
                    morphemes.append(root)
                for suffix in word_analysis["ekler"]:
                    suffix_clean = suffix.lstrip('-')
                    if suffix_clean:
                        morphemes.append(suffix_clean)
            
            if morphemes:
                processed_lines.append(' '.join(morphemes))
        except Exception as e:
            # Hatalı satırları atla ama logla
            # print(f"Error processing line in process {os.getpid()}: {line[:50]}... | Error: {e}")
            pass
    return processed_lines

def main():
    parser = argparse.ArgumentParser(description="Preprocesses a corpus in parallel using morphological analysis.")
    parser.add_argument("--input", required=True, help="Path to the cleaned corpus file.")
    parser.add_argument("--output", required=True, help="Path to save the morpho-processed corpus file.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: all available cores).")
    parser.add_argument("--chunksize", type=int, default=1000, help="Number of lines to process in each chunk per worker.")
    args = parser.parse_args()

    num_workers = args.workers if args.workers else mp.cpu_count()
    print(f"🚀 Starting parallel preprocessing with {num_workers} workers.")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")

    # Get total line count for tqdm progress bar without loading the whole file
    print("Counting lines in input file...")
    with open(args.input, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out, \
         mp.Pool(processes=num_workers, initializer=init_worker) as pool:

        # Process the file in chunks using a generator to avoid loading all lines into memory
        def chunk_generator(file, size):
            chunk = []
            for line in file:
                chunk.append(line)
                if len(chunk) == size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

        results_iterator = pool.imap_unordered(process_chunk, chunk_generator(f_in, args.chunksize))
        
        progress_bar = tqdm(total=total_lines, desc="Parallel Preprocessing", unit="lines")
        
        lines_processed = 0
        for processed_chunk in results_iterator:
            if processed_chunk:
                f_out.write('\n'.join(processed_chunk) + '\n')
                lines_processed += len(processed_chunk)
                progress_bar.update(len(processed_chunk))
            
    progress_bar.close()
    print("\n✅ Parallel preprocessing complete!")
    print(f"   Processed corpus saved to: {args.output}")

if __name__ == "__main__":
    main()