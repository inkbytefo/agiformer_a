# agiformer/language/morpho_splitter.py

# Developer: inkbytefo
# Modified: 2025-11-06

import logging
import zeyrek

logger = logging.getLogger(__name__)

class ZeyrekSplitter:
    """
    A wrapper for the Zeyrek morphological analyzer. This class replaces
    the previous implementations to provide a lightweight and performant
    alternative for Turkish morphological analysis, avoiding heavy dependencies
    like TensorFlow.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing ZeyrekSplitter singleton...")
            cls._instance = super(ZeyrekSplitter, cls).__new__(cls)
            try:
                # Zeyrek's MorphAnalyzer is lightweight and initializes quickly.
                cls._instance.analyzer = zeyrek.MorphAnalyzer()
                logger.info("Zeyrek.MorphAnalyzer instance created successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Zeyrek.MorphAnalyzer: {e}", exc_info=True)
                cls._instance = None
                raise
        return cls._instance

    def split_sentence(self, sentence: str) -> dict:
        """
        Analyzes a sentence and splits each word into its root and morphemes
        using Zeyrek.

        Args:
            sentence: The input sentence string.

        Returns:
            A dictionary containing the analysis results, structured for
            compatibility with the preprocessing scripts.
        """
        if not hasattr(self, 'analyzer'):
            raise RuntimeError("ZeyrekSplitter is not properly initialized.")

        # Zeyrek analyzes word by word.
        words = sentence.split()
        output = {"kelimeler": []}

        for word in words:
            # Perform analysis for the word
            analysis_results = self.analyzer.analyze(word)
            
            # The first result is typically the most probable one.
            best_analysis = analysis_results[0][0] if analysis_results and analysis_results[0] else None

            if best_analysis:
                root = best_analysis.lemma
                # Get morphemes, excluding the root itself.
                suffixes = [m for m in best_analysis.morphemes if m != root and m != 'Unk']
                
                word_dict = {
                    "kök": root,
                    "ekler": suffixes
                }
                output["kelimeler"].append(word_dict)
            else:
                # If no analysis, treat the word as its own root.
                output["kelimeler"].append({
                    "kök": word,
                    "ekler": []
                })
                
        return output
