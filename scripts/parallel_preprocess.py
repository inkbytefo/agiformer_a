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

from agiformer.language.morpho_splitter import MorphoSplitter

# Global MorphoSplitter nesnesi (her bir alt işlemde bir kez başlatılacak)
splitter = None

def init_worker():
    """Her bir alt işlem (worker) için MorphoSplitter'ı başlatır."""
    global splitter
    print(f"Initializing MorphoSplitter for process {os.getpid()}...")
    splitter = MorphoSplitter()

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

    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out, \
         mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        
        lines = f_in.readlines()
        total_lines = len(lines)
        
        # Veriyi parçalara ayır ve havuzda işle
        # `imap_unordered` sonuçları geldikçe işler, bu da bellek kullanımını azaltır
        # ve ilerlemeyi daha iyi gösterir.
        results_iterator = pool.imap_unordered(process_chunk, [lines[i:i + args.chunksize] for i in range(0, total_lines, args.chunksize)])
        
        progress_bar = tqdm(total=total_lines, desc="Parallel Preprocessing")
        
        for processed_chunk in results_iterator:
            for line in processed_chunk:
                f_out.write(line + '\n')
            progress_bar.update(len(processed_chunk)) # Gerçek işlenen satır sayısı kadar güncelle
            
    progress_bar.close()
    print("\n✅ Parallel preprocessing complete!")
    print(f"   Processed corpus saved to: {args.output}")

if __name__ == "__main__":
    main()