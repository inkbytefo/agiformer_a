#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
============================================================================
Morfem Ayrımı - Zemberek Entegrasyonu
Türkçe kelimeleri kök + ek olarak ayırır
============================================================================
"""

import json
import re
from typing import List, Dict, Optional, Tuple
import subprocess
import os

class MorphoSplitter:
    """Türkçe morfem ayrımı için Zemberek entegrasyonu"""
    
    def __init__(self, use_java: bool = True):
        """
        MorphoSplitter başlat
        
        Args:
            use_java: Zemberek Java kullan (True) veya basit regex (False)
        """
        self.use_java = use_java
        self.java_available = self._check_java()
        
        if not self.java_available and use_java:
            print("⚠️  Java not found. Using simple regex-based morphological analysis.")
            self.use_java = False
        
        # Basit ekler sözlüğü (fallback) - tirelerle birlikte
        self.common_suffixes = [
            # Çoklu ek kombinasyonları (öncelik: uzun olanlar)
            '-lerimdeki', '-lerimde', '-lerim',  # ev-ler-im-de-ki-ler
            '-lerinden', '-lerine', '-lerini', '-lerinde', '-lerindeki',
            '-iler', '-ler', '-lar',  # Son -ler/-lar sonundaki ek
            '-leri', '-ları', '-yi', '-yı',  # Belirtme hal eki sonrası
            '-yu', '-yü',  # Belirtme hal eki (buffer y) ek varyantları
            
            # Tekli ekler
            '-ler', '-lar',  # Çoğul
            '-im', '-ım', '-um', '-üm',  # İyelik 1. tekil
            '-ın', '-in', '-un', '-ün',  # İyelik 2. tekil
            '-i', '-ı', '-u', '-ü',  # İyelik 3. tekil / Belirtme
            '-imiz', '-ımız', '-umuz', '-ümüz',  # İyelik 1. çoğul
            '-iniz', '-ınız', '-unuz', '-ünüz',  # İyelik 2. çoğul
            '-leri', '-ları',  # İyelik 3. çoğul
            '-de', '-da', '-te', '-ta',  # Bulunma
            '-den', '-dan', '-ten', '-tan',  # Ayrılma
            '-e', '-a',  # Yönelme
            '-in', '-ın', '-ün', '-un',  # İlgi
            '-ki',  # İlgi eki
            '-dir', '-dır', '-tir', '-tır', '-dur', '-dür', '-tur', '-tür',  # Ek-fiil
            '-miş', '-mış', '-muş', '-müş',  # Geçmiş zaman
            '-mışım', '-mışsın', '-mış', '-mışız', '-mışsınız', '-mışlar',
            '-yor', '-yorsun', '-yor', '-yoruz', '-yorsunuz', '-yorlar',  # Şimdiki zaman
            '-acak', '-ecek',  # Gelecek zaman
            '-acağım', '-eceğim', '-acaksın', '-eceksin', '-acak', '-ecek',
            '-dı', '-di', '-du', '-dü', '-tı', '-ti', '-tu', '-tü',  # Görülen geçmiş
            '-dım', '-dim', '-dum', '-düm', '-tım', '-tim', '-tum', '-tüm',
            # Bağlaç ve diğer
            '-yle', '-yla', '-ile', '-la', '-le',
            '-ken',  # -iken kısa formu
            '-mek', '-mak',  # Mastar
        ]
        
        # Ünlü uyumu kuralları
        self.vowel_harmony = {
            'back_vowels': {'a', 'ı', 'o', 'u'},
            'front_vowels': {'e', 'i', 'ö', 'ü'}
        }
    
    def _check_java(self) -> bool:
        """Java'nın yüklü olup olmadığını kontrol et"""
        try:
            result = subprocess.run(['java', '-version'], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=2)
            return result.returncode == 0
        except:
            return False
    
    def _zemberek_analyze(self, word: str) -> Optional[List[Dict]]:
        """
        Zemberek ile morfolojik analiz (Python binding)
        GitHub: https://github.com/ahmetaa/zemberek-nlp
        
        Args:
            word: Analiz edilecek kelime
        
        Returns:
            Morfem listesi veya None
        """
        try:
            # Try zemberek-nlp Python package (v0.17.1)
            import zemberek
            from zemberek import TurkishMorphology
            
            # Initialize morphology (singleton pattern for performance)
            if not hasattr(self, '_morphology'):
                self._morphology = TurkishMorphology.create_with_defaults()
            
            # Analyze word - returns list of SingleAnalysis objects
            results = self._morphology.analyze(word)
            
            if not results or len(results) == 0:
                return None
            
            # Get first (most likely) analysis
            analysis = results[0]
            
            # Extract morphemes using proper Zemberek API
            morphemes = []
            
            # Get lemma (dictionary form/root)
            lemma = analysis.get_lemma()
            pos_tag = str(analysis.get_pos()) if analysis.get_pos() else "Unknown"
            
            if lemma:
                morphemes.append({
                    "morfem": lemma,
                    "tür": self._classify_pos_tag(pos_tag),
                    "pozisyon": 0
                })
            
            # Get morpheme data using proper API methods
            try:
                # Try to get morpheme data directly
                morpheme_data = analysis.get_morpheme_data()
                if morpheme_data:
                    for i, morph_data in enumerate(morpheme_data[1:], 1):  # Skip root
                        morpheme_surface = str(morph_data.morpheme)
                        morpheme_type = str(morph_data.morpheme.morpheme_type) if hasattr(morph_data.morpheme, 'morpheme_type') else "Unknown"
                        
                        morphemes.append({
                            "morfem": morpheme_surface,
                            "tür": self._classify_zemberek_suffix(morpheme_type),
                            "pozisyon": i
                        })
                else:
                    # Fallback: parse from string representation
                    self._parse_zemberek_string_analysis(analysis, morphemes, lemma)
                    
            except (AttributeError, Exception):
                # Fallback: parse from string representation
                self._parse_zemberek_string_analysis(analysis, morphemes, lemma)
            
            # Final fallback: if no suffixes found but word != lemma
            if len(morphemes) == 1 and word != lemma and word.startswith(lemma):
                suffix = word[len(lemma):]
                if suffix:
                    morphemes.append({
                        "morfem": suffix,
                        "tür": "ek",
                        "pozisyon": 1
                    })
            
            return morphemes if len(morphemes) > 0 else None
        
        except ImportError:
            # zemberek-nlp not installed
            return None
        except Exception as e:
            # Fallback to regex if Zemberek fails
            return None
    
    def _parse_zemberek_string_analysis(self, analysis, morphemes: List[Dict], lemma: str):
        """Zemberek string çıktısından morfem bilgilerini çıkar"""
        try:
            # Get string representation of analysis
            morphological_analysis = str(analysis)
            
            # Parse morphemes from analysis string
            # Format examples: [ev:Noun][ler:A3pl][im:P1sg][de:Loc][ki:Rel][ler:A3pl]
            if '[' in morphological_analysis and ']' in morphological_analysis:
                # Extract all bracketed parts
                import re
                parts = re.findall(r'\[([^\]]+)\]', morphological_analysis)
                
                for i, part in enumerate(parts[1:], 1):  # Skip first part (lemma)
                    if ':' in part:
                        surface, morph_type = part.split(':', 1)
                        if surface and surface != lemma:
                            morphemes.append({
                                "morfem": surface,
                                "tür": self._classify_zemberek_suffix(morph_type),
                                "pozisyon": i
                            })
                    elif part and part != lemma:
                        morphemes.append({
                            "morfem": part,
                            "tür": "ek",
                            "pozisyon": i
                        })
        except Exception:
            pass
    
    def _classify_pos_tag(self, pos_tag: str) -> str:
        """POS etiketini Türkçe sınıfa çevir"""
        pos_lower = pos_tag.lower()
        if 'noun' in pos_lower:
            return "isim_kök"
        elif 'verb' in pos_lower:
            return "fiil_kök"
        elif 'adj' in pos_lower:
            return "sıfat_kök"
        elif 'adv' in pos_lower:
            return "zarf_kök"
        else:
            return "kök"
    
    def _classify_zemberek_suffix(self, suffix_info: str) -> str:
        """
        Zemberek suffix bilgisini sınıflandır
        Zemberek v0.17.1 morfem türlerine göre güncellenmiş
        """
        suffix_lower = suffix_info.lower()
        
        # Çoğul ekleri
        if 'a3pl' in suffix_lower or 'plural' in suffix_lower:
            return "çoğul"
        
        # İyelik ekleri (Possessive)
        elif any(x in suffix_lower for x in ['poss', 'p1sg', 'p2sg', 'p3sg', 'p1pl', 'p2pl', 'p3pl']):
            return "iyelik"
        
        # Durum ekleri (Case)
        elif 'loc' in suffix_lower:
            return "bulunma"
        elif 'abl' in suffix_lower:
            return "ayrılma"
        elif 'dat' in suffix_lower:
            return "yönelme"
        elif 'gen' in suffix_lower:
            return "ilgi"
        elif 'acc' in suffix_lower:
            return "belirtme"
        elif 'ins' in suffix_lower or 'instr' in suffix_lower:
            return "araç"
        
        # Zaman ekleri (Tense)
        elif any(x in suffix_lower for x in ['past', 'pret', 'narr']):
            return "geçmiş_zaman"
        elif any(x in suffix_lower for x in ['pres', 'prog', 'cont']):
            return "şimdiki_zaman"
        elif 'fut' in suffix_lower:
            return "gelecek_zaman"
        elif 'aor' in suffix_lower:
            return "geniş_zaman"
        elif 'opt' in suffix_lower:
            return "istek"
        elif 'imp' in suffix_lower:
            return "emir"
        elif 'cond' in suffix_lower:
            return "şart"
        
        # Kişi ekleri (Person)
        elif any(x in suffix_lower for x in ['a1sg', 'a2sg', 'a3sg', 'a1pl', 'a2pl', 'a3pl']):
            return "kişi"
        
        # İlişki ekleri
        elif 'rel' in suffix_lower:
            return "ilgi"
        elif 'with' in suffix_lower:
            return "birliktelik"
        
        # Fiil türetme ekleri
        elif any(x in suffix_lower for x in ['caus', 'causative']):
            return "ettirgen"
        elif any(x in suffix_lower for x in ['pass', 'passive']):
            return "edilgen"
        elif any(x in suffix_lower for x in ['reflex', 'reflexive']):
            return "dönüşlü"
        elif any(x in suffix_lower for x in ['recip', 'reciprocal']):
            return "işteş"
        
        # İsim türetme ekleri
        elif any(x in suffix_lower for x in ['agt', 'agent']):
            return "fail"
        elif any(x in suffix_lower for x in ['dim', 'diminutive']):
            return "küçültme"
        
        # Diğer özel durumlar
        elif 'ness' in suffix_lower:
            return "soyut_isim"
        elif 'ly' in suffix_lower or 'adv' in suffix_lower:
            return "zarf_yapım"
        elif 'adj' in suffix_lower:
            return "sıfat_yapım"
        
        # Varsayılan
        else:
            return "ek"
    
    def _regex_analyze(self, word: str) -> List[Dict]:
        """
        Regex tabanlı gelişmiş morfem ayrımı (çoklu ek desteği)
        
        Args:
            word: Analiz edilecek kelime
        
        Returns:
            Morfem listesi
        """
        morphemes = []
        # Normalize: lower-case, keep apostrophe for proper noun suffix handling
        remaining = word.lower()
        
        # Özel durum: Apostrof sonrası yönelme eki (örn. "Ankara'ya")
        if "'" in remaining and (remaining.endswith("'ya") or remaining.endswith("'ye")):
            base = remaining.split("'")[0]
            cmp_suffix = 'ya' if remaining.endswith("'ya") else 'ye'
            morphemes = [
                {"morfem": base, "tür": "kök", "pozisyon": 0},
                {"morfem": cmp_suffix, "tür": "yönelme", "pozisyon": 1},
            ]
            return morphemes

        # İyileştirme: Çoklu ek desteği (iteratif ayrım)
        max_iterations = 10  # Sonsuz döngü önleme
        iteration = 0
        
        while remaining and iteration < max_iterations:
            iteration += 1
            suffix_found = False
            
            # En uzun eklerden başlayarak geriye doğru
            sorted_suffixes = sorted(self.common_suffixes, key=len, reverse=True)
            
            for suffix in sorted_suffixes:
                # Compare using stripped hyphen
                cmp_suffix = suffix.lstrip('-')
                if remaining.endswith(cmp_suffix):
                    # Ek bulundu
                    root = remaining[:-len(cmp_suffix)]
                    
                    # Minimum kök uzunluğu kontrolü
                    if len(root) >= 2:
                        # Ekleri başa ekle (ters sıra olacak, düzelteceğiz)
                        morphemes.insert(0, {
                            "morfem": cmp_suffix,
                            "tür": self._classify_suffix(suffix),
                            "pozisyon": len(morphemes) + 1
                        })
                        remaining = root
                        suffix_found = True
                        break

            # Özel durum: özel isimlerde yönelme eki ("Ankara'ya", "İzmir'ye")
            if not suffix_found and ("'ya" in remaining or "'ye" in remaining):
                if remaining.endswith("'ya") or remaining.endswith("'ye"):
                    cmp_suffix = 'ya' if remaining.endswith("'ya") else 'ye'
                    root = remaining[:-len("'" + cmp_suffix)]
                    # Apostrof kökten kaldır
                    if root.endswith("'"):
                        root = root[:-1]
                    if len(root) >= 2:
                        morphemes.insert(0, {
                            "morfem": cmp_suffix,
                            "tür": "yönelme",
                            "pozisyon": len(morphemes) + 1
                        })
                        remaining = root
                        suffix_found = True
            
            # Ek bulunamadıysa dur
            if not suffix_found:
                break
        
        # Kökü ekle
        if remaining:
            morphemes.insert(0, {
                "morfem": remaining,
                "tür": "kök",
                "pozisyon": 0
            })
        
        # Pozisyon indekslerini düzelt
        for i, morf in enumerate(morphemes):
            morf["pozisyon"] = i
        
        return morphemes
    
    def _classify_suffix(self, suffix: str) -> str:
        """Eki sınıflandır"""
        suffix_lower = suffix.lower()
        suffix_lower = suffix_lower if suffix_lower.startswith('-') else f'-{suffix_lower}'
        
        if suffix_lower in ['-ler', '-lar']:
            return "çoğul"
        elif suffix_lower in ['-im', '-ın', '-i', '-imiz', '-iniz', '-leri']:
            return "iyelik"
        elif suffix_lower in ['-de', '-da', '-te', '-ta']:
            return "bulunma"
        elif suffix_lower in ['-den', '-dan', '-ten', '-tan']:
            return "ayrılma"
        elif suffix_lower in ['-e', '-a']:
            return "yönelme"
        elif suffix_lower in ['-i', '-ı', '-u', '-ü', '-yi', '-yı', '-yu', '-yü']:
            return "belirtme"
        elif suffix_lower == '-ki':
            return "ilgi"
        elif suffix_lower in ['-miş', '-mış', '-muş', '-müş']:
            return "geçmiş_zaman"
        elif suffix_lower == '-yor':
            return "şimdiki_zaman"
        elif suffix_lower in ['-acak', '-ecek']:
            return "gelecek_zaman"
        elif suffix_lower in ['-dı', '-di', '-du', '-dü', '-tı', '-ti', '-tu', '-tü']:
            return "görülen_geçmiş"
        elif suffix_lower in ['-dir', '-dır', '-tir', '-tır']:
            return "ek_fiil"
        else:
            return "diğer_ek"
    
    def split_word(self, word: str) -> Dict:
        """
        Kelimeyi morfemlere ayır
        
        Args:
            word: Analiz edilecek kelime
        
        Returns:
            Morfem analizi dict'i
        """
        if not word or len(word.strip()) == 0:
            return {
                "kelime": word,
                "morfemler": [],
                "kök": "",
                "ekler": []
            }
        
        word_clean = word.strip()
        
        # Zemberek kullanılabilirse önce onu dene
        if self.use_java and self.java_available:
            zemberek_result = self._zemberek_analyze(word_clean)
            if zemberek_result:
                return {
                    "kelime": word_clean,
                    "morfemler": zemberek_result,
                    "kök": zemberek_result[0]["morfem"] if zemberek_result else word_clean,
                    "ekler": [m["morfem"] for m in zemberek_result[1:]] if len(zemberek_result) > 1 else []
                }
        
        # Regex fallback
        morphemes = self._regex_analyze(word_clean)
        
        root = morphemes[0]["morfem"] if morphemes and morphemes[0]["tür"] == "kök" else word_clean
        suffixes = [m["morfem"] for m in morphemes[1:]] if len(morphemes) > 1 else []
        
        return {
            "kelime": word_clean,
            "morfemler": morphemes,
            "kök": root,
            "ekler": suffixes
        }
    
    def split_sentence(self, sentence: str) -> Dict:
        """
        Cümleyi kelimelere ayırıp her kelimeyi morfemlere böl
        
        Args:
            sentence: Analiz edilecek cümle
        
        Returns:
            Cümle morfem analizi
        """
        # Kelimelere ayır
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        sentence_analysis = {
            "cümle": sentence,
            "kelimeler": [],
            "tüm_morfemler": []
        }
        
        for word in words:
            word_analysis = self.split_word(word)
            sentence_analysis["kelimeler"].append(word_analysis)
            sentence_analysis["tüm_morfemler"].extend(word_analysis["morfemler"])
        
        return sentence_analysis
    
    def to_json(self, analysis: Dict, pretty: bool = True) -> str:
        """Analizi JSON formatına çevir"""
        if pretty:
            return json.dumps(analysis, ensure_ascii=False, indent=2)
        else:
            return json.dumps(analysis, ensure_ascii=False)
    
    def is_valid_vowel_harmony(self, root: str, suffix: str) -> bool:
        """
        Ünlü uyumu kontrolü
        
        Args:
            root: Kök kelime
            suffix: Ek
        
        Returns:
            Ünlü uyumu doğru mu?
        """
        if not root or not suffix:
            return True
        
        # Kökün son ünlüsünü bul
        root_vowels = [c for c in root if c in 'aeıiouöü']
        if not root_vowels:
            return True
        
        last_vowel = root_vowels[-1]
        
        # Ekin ilk ünlüsünü bul
        suffix_vowels = [c for c in suffix if c in 'aeıiouöü']
        if not suffix_vowels:
            return True
        
        first_vowel = suffix_vowels[0]
        
        # Ünlü uyumu kuralları
        back_vowels = self.vowel_harmony['back_vowels']
        front_vowels = self.vowel_harmony['front_vowels']
        
        # Kalın-ince uyumu
        if last_vowel in back_vowels:
            return first_vowel in back_vowels or first_vowel in {'a', 'ı', 'u'}
        elif last_vowel in front_vowels:
            return first_vowel in front_vowels or first_vowel in {'e', 'i', 'ü'}
        
        return True

def main():
    """Test fonksiyonu"""
    splitter = MorphoSplitter()
    
    test_words = [
        "Evlerimdekiler",
        "Gittim",
        "Anladım",
        "Marketten",
        "Okuldan",
        "Dünü",
        "Yaptın"
    ]
    
    print("\n" + "=" * 60)
    print("    Morfem Ayrımı Test")
    print("=" * 60)
    
    for word in test_words:
        result = splitter.split_word(word)
        print(f"\n📝 Kelime: {word}")
        print(f"   Kök: {result['kök']}")
        print(f"   Ekler: {', '.join(result['ekler']) if result['ekler'] else '(yok)'}")
        print(f"   Morfemler:")
        for morf in result['morfemler']:
            print(f"      - {morf['morfem']} ({morf['tür']})")
    
    # Cümle analizi
    print("\n" + "=" * 60)
    sentence = "Dün markete gittim"
    result = splitter.split_sentence(sentence)
    print(f"\n📄 Cümle: {sentence}")
    print(splitter.to_json(result))

if __name__ == "__main__":
    main()

