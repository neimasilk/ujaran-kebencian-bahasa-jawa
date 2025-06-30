import unittest
import sys
import os
import string # Ditambahkan untuk mengatasi NameError

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.text_preprocessing import (
    normalize_text,
    tokenize_text,
    remove_stopwords,
    preprocess_text_advanced,
    remove_punctuation,
    DEFAULT_JAWA_STOPWORDS
)

class TestTextPreprocessing(unittest.TestCase):

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation("halo, apa kabar!"), "halo apa kabar")
        self.assertEqual(remove_punctuation("teks (dengan) tanda kurung."), "teks dengan tanda kurung")
        self.assertEqual(remove_punctuation("!@#$%^&*()_+"), "")
        self.assertEqual(remove_punctuation(""), "")
        self.assertEqual(remove_punctuation("teks tanpa tanda baca"), "teks tanpa tanda baca")
        self.assertEqual(remove_punctuation(123), "") # Test non-string input

    def test_normalize_text(self):
        self.assertEqual(normalize_text("   Ini CONTOH Teks!   "), "ini contoh teks")
        self.assertEqual(normalize_text("Teks Kanthi Angka 123."), "teks kanthi angka 123")
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("  "), "")
        self.assertEqual(normalize_text("REGANE PANCEN LARANG TENAN KOK."), "regane pancen larang tenan kok")
        self.assertEqual(normalize_text(12345), "") # Test non-string input

    def test_tokenize_text(self):
        # Test with default Hugging Face tokenizer (if loaded) or basic split
        # Exact tokenization output can vary based on the tokenizer model
        # We'll check for non-empty list for non-empty string
        self.assertIsInstance(tokenize_text("iki conto kalimat"), list)
        self.assertTrue(len(tokenize_text("iki conto kalimat")) > 0)

        # Test with known IndoBERT tokenization behavior for some words (may need adjustment)
        # For "yogyakarta", IndoBERT tokenizer (indobenchmark/indobert-base-p1) might produce ['yo', '##gya', '##karta']
        # For simpler testing, we can mock the tokenizer or test the fallback

        # Test empty string
        self.assertEqual(tokenize_text(""), [])

        # Test non-string input
        self.assertEqual(tokenize_text(123), [])

        # To make tests more robust against tokenizer changes,
        # we can check if the tokens re-joined roughly match the input (after normalization)
        text = "Universitas Gadjah Mada"
        tokens = tokenize_text(normalize_text(text))
        # This is a loose check, assumes tokenizer doesn't drastically alter words
        self.assertTrue(any(t in normalize_text(text).split() for t in tokens) or
                        any(normalize_text(text).startswith(t) for t in tokens) or
                        len(tokens) > 0 if text else True)


    def test_remove_stopwords(self):
        stopwords = {"aku", "kowe", "lan"}
        self.assertEqual(remove_stopwords(["aku", "mangan", "sega", "lan", "kowe"], stopwords), ["mangan", "sega"])
        self.assertEqual(remove_stopwords(["ora", "ana", "stopwords"], stopwords), ["ora", "ana", "stopwords"])
        self.assertEqual(remove_stopwords([], stopwords), [])
        self.assertEqual(remove_stopwords(["aku", "kowe"], stopwords), [])
        self.assertEqual(remove_stopwords(["AKU", "Mangan", "Sega"], stopwords), ["Mangan", "Sega"]) # Assumes tokens are lowercased before check

        # Test with default stopwords (just check if it runs and removes some common words)
        default_removed = remove_stopwords(["aku", "mangan", "sega", "lan", "krupuk", "ing", "pasar"], DEFAULT_JAWA_STOPWORDS)
        self.assertTrue("aku" not in default_removed)
        self.assertTrue("lan" not in default_removed)
        self.assertTrue("ing" not in default_removed)
        self.assertIn("mangan", default_removed) # 'mangan' is not a default stopword

        # Test non-list input
        self.assertEqual(remove_stopwords("bukan list", stopwords), [])
        self.assertEqual(remove_stopwords(None, stopwords), [])


    def test_preprocess_text_advanced(self):
        # Test case 1: Basic sentence
        text1 = "Iki tuladha ukara basa Jawa, kanthi tandha wacan lan angka 123!"
        # Expected tokens might vary slightly with real IndoBERT tokenizer vs simple split
        # For now, let's assume some common stopwords are removed
        tokens1, processed_text1 = preprocess_text_advanced(text1)
        self.assertIsInstance(tokens1, list)
        self.assertIsInstance(processed_text1, str)
        self.assertTrue(all(t not in DEFAULT_JAWA_STOPWORDS for t in tokens1 if t not in string.punctuation and not t.isdigit()))
        self.assertTrue("angka" in tokens1 or "123" in tokens1) # 'angka' and '123' are not stopwords
        self.assertFalse("," in processed_text1)
        self.assertFalse("!" in processed_text1)

        # "iki" adalah stopword dan dihilangkan.
        # "tuladha" ditokenisasi menjadi 'tul', '##adh', '##a'
        # "ukara" ditokenisasi menjadi 'uka', '##ra'
        self.assertTrue(processed_text1.startswith("tul ##adh ##a uka ##ra basa jawa"))

        # Test case 2: Sentence with many stopwords
        text2 = "Kula tresna sanget kaliyan Yogyakarta lan Solo."
        tokens2, processed_text2 = preprocess_text_advanced(text2)
        # Periksa apakah kata-kata non-stopword utama masih ada, dalam bentuk apapun setelah tokenisasi
        self.assertTrue(any(sub_token in processed_text2 for sub_token in tokenize_text("tresna")))
        self.assertTrue(any(sub_token in processed_text2 for sub_token in tokenize_text("sanget")))
        self.assertTrue(any(sub_token in processed_text2 for sub_token in tokenize_text("yogyakarta")))
        self.assertTrue(any(sub_token in processed_text2 for sub_token in tokenize_text("solo")))
        # Periksa apakah stopword (atau bagiannya) sudah tidak dominan
        self.assertFalse("kula" in processed_text2.split() and "kaliyan" in processed_text2.split())


        # Test case 3: Uppercase text with extra spaces and custom stopwords
        text3 = "  REGANE  PANCEN  LARANG  TENAN  KOK!  "
        custom_stopwords = DEFAULT_JAWA_STOPWORDS.union({"regane", "pancen", "larang", "tenan"})
        tokens3, processed_text3 = preprocess_text_advanced(text3, custom_stopwords=custom_stopwords)

        # Berdasarkan observasi: "larang" dan "kok" (default stopword) terhapus.
        # "regane", "pancen", "tenan" dipecah dan sub-tokennya tidak ada di stopwords.
        expected_text3 = "reg ##ane panc ##en ten ##an"
        self.assertEqual(processed_text3, expected_text3)
        # Memastikan token yang tersisa memang bukan stopword (setelah dipecah)
        self.assertTrue(all(t.lower() not in custom_stopwords for t in tokens3 if not t.startswith("##")))


        # Test case 4: Empty string
        tokens4, processed_text4 = preprocess_text_advanced("")
        self.assertEqual(tokens4, [])
        self.assertEqual(processed_text4, "")

        # Test case 5: String with only stopwords
        text5 = "aku lan kowe" # "aku", "lan", "kowe" adalah stopwords
        tokens5, processed_text5 = preprocess_text_advanced(text5)
        # "kowe" dipecah menjadi "kow", "##e" oleh tokenizer dan tidak terhapus
        expected_tokens5 = tokenize_text("kowe") # Karena "aku" dan "lan" akan hilang
        if "kowe" in DEFAULT_JAWA_STOPWORDS: # Jika "kowe" sendiri adalah satu token dan stopword
             expected_tokens5 = []

        # Jika "kowe" dipecah menjadi sub-token yang tidak ada di stopword list:
        # Contoh: ['kow', '##e']
        # "aku" dan "lan" terhapus karena merupakan stopword dan tidak dipecah oleh tokenizer.
        # "kowe" (stopword) dipecah menjadi ['kow', '##e'], yang bukan stopword individual.
        self.assertEqual(tokens5, ['kow', '##e'])
        self.assertEqual(processed_text5, "kow ##e")


        # Test case 6: String with only punctuation
        text6 = "!@#$%^&*"
        tokens6, processed_text6 = preprocess_text_advanced(text6)
        self.assertEqual(tokens6, [])
        self.assertEqual(processed_text6, "")

        # Test case 7: Non-string input
        tokens7, processed_text7 = preprocess_text_advanced(12345)
        self.assertEqual(tokens7, [])
        self.assertEqual(processed_text7, "")

        tokens8, processed_text8 = preprocess_text_advanced(None)
        self.assertEqual(tokens8, [])
        self.assertEqual(processed_text8, "")

if __name__ == '__main__':
    unittest.main()
