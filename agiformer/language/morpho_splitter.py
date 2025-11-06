# agiformer/language/morpho_splitter.py

# Developer: inkbytefo
# Modified: 2025-11-06

import logging
from vngrs_nlp.morphology import VngrsMorphology

logger = logging.getLogger(__name__)

class VnlpSplitter:
    """
    A wrapper for the VNLP morphological analyzer to provide a consistent
    interface for splitting sentences into their morphemes (roots and suffixes).
    This class replaces the previous Zemberek-based implementation for better
    performance and modern features.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing VnlpSplitter singleton...")
            cls._instance = super(VnlpSplitter, cls).__new__(cls)
            try:
                cls._instance.analyzer = VngrsMorphology()
                logger.info("VngrsMorphology instance created successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize VngrsMorphology: {e}", exc_info=True)
                cls._instance = None
                raise
        return cls._instance

    def split_sentence(self, sentence: str) -> dict:
        """
        Analyzes a sentence and splits each word into its root and morphemes.

        Args:
            sentence: The input sentence string.

        Returns:
            A dictionary containing the analysis results, structured for
            compatibility with the preprocessing scripts.
        """
        if not hasattr(self, 'analyzer'):
            raise RuntimeError("VnlpSplitter is not properly initialized.")

        analysis_results = self.analyzer.analyze(sentence)
        
        output = {"kelimeler": []}
        
        for word_analysis in analysis_results:
            # The first analysis is usually the most likely one
            best_analysis = word_analysis[0] if word_analysis else None
            
            if best_analysis:
                root = best_analysis.get_root()
                morphemes = best_analysis.get_morphemes()
                
                # VNLP provides morphemes including the root, so we extract suffixes
                suffixes = [m for m in morphemes if m != root]
                
                word_dict = {
                    "kök": root,
                    "ekler": suffixes
                }
                output["kelimeler"].append(word_dict)
            else:
                # If no analysis, treat the word as its own root
                # This handles punctuation, numbers, or unknown words gracefully
                output["kelimeler"].append({
                    "kök": word_analysis.get_surface() if hasattr(word_analysis, 'get_surface') else "",
                    "ekler": []
                })
                
        return output
