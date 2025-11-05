# Developer: inkbytefo
# Modified: 2025-11-05

import argparse
from tqdm import tqdm
import re

def clean_line(line: str) -> str:
    """Tek bir satırı temizler."""
    # Null byte'ları kaldır
    line = line.replace('\x00', '')
    
    # Film sitesi menüleri gibi tekrarlayan ve anlamsız kalıpları kaldır
    # Bu regex'ler loglardaki örneklere göre hazırlandı, genişletilebilir.
    patterns_to_remove = [
        r'►.*?Filmleri',
        r'❖ENGLISH SUBTITLES.*',
        r'⇨ ENGLISH SUBTITLES.*',
        r'▂ ▃ ▅ ▆ █.*?█ ▆ ▅ ▃ ▂',
        r'IMDb Top 250',
        r'YAPIM YILI:',
        r'ÜLKE:',
        r'ÜYE YORUMU:',
        r'izle,izle,',
        r'full izle,'
    ]
    for pattern in patterns_to_remove:
        line = re.sub(pattern, '', line, flags=re.IGNORECASE)
        
    # Çoklu boşlukları tek boşluğa indir
    line = re.sub(r'\s+', ' ', line).strip()
    
    return line

def main():
    parser = argparse.ArgumentParser(description="Cleans a raw text corpus.")
    parser.add_argument("--input", required=True, help="Path to the raw corpus file.")
    parser.add_argument("--output", required=True, help="Path to save the cleaned corpus file.")
    parser.add_argument("--min-len", type=int, default=20, help="Minimum line length in characters.")
    parser.add_argument("--max-len", type=int, default=4000, help="Maximum line length in characters.")
    args = parser.parse_args()

    print(f"Cleaning corpus: {args.input}")
    
    lines_read = 0
    lines_written = 0
    
    with open(args.input, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Cleaning lines"):
            lines_read += 1
            cleaned_line = clean_line(line)
            
            # Uzunluk kontrolleri
            if args.min_len <= len(cleaned_line) <= args.max_len:
                f_out.write(cleaned_line + '\n')
                lines_written += 1

    print("\n✅ Cleaning complete!")
    print(f"   Lines read: {lines_read:,}")
    print(f"   Lines written: {lines_written:,} ({lines_written/lines_read:.2%})")
    print(f"   Cleaned corpus saved to: {args.output}")

if __name__ == "__main__":
    main()