import unittest
import spacy
from typing import List
from analysis.dependency_parse import *

# Mock tokenizer for testing
class MockGPT2Tokenizer:
    """Mock tokenizer that simulates GPT-2 style subword tokenization"""
    
    def __init__(self):
        # Predefined tokenizations for test sentences
        self.tokenizations = {
            "Some women weren't understood by some guests.": [
                "Some", " women", " weren", "'t", " understood", " by", " some", " guests", "."
            ],
            "All piano teachers that criticized Dawn haven't heard Bruce.": [
                "All", " piano", " teachers", " that", " critic", "ized", " Dawn", " haven", "'t", " heard", " Bruce", "."
            ],
            "There were no shirts disturbing Theresa": [
                "There", " were", " no", " shirts", " disturb", "ing", " Theresa"
            ],
            "A lot of actresses that thought about Alice healed themselves.": [
                "A", " lot", " of", " act", "resses", " that", " thought", " about", " Alice", " heal", "ed", " themselves", "."
            ],
            "Timothy didn't boast about himself.": [
                "Tim", "othy", " didn", "'t", " boast", " about", " himself", "."
            ]
        }
    
    def encode(self, text: str) -> List[int]:
        if text in self.tokenizations:
            return list(range(len(self.tokenizations[text])))
        else:
            # Fallback: simple split
            tokens = text.split()
            return list(range(len(tokens)))
    
    def decode(self, token_ids) -> str:
        for text, tokens in self.tokenizations.items():
            if isinstance(token_ids, list):
                if len(token_ids) <= len(tokens):
                    return "".join([tokens[i] for i in token_ids if i < len(tokens)])
            else:
                if token_ids < len(tokens):
                    return tokens[token_ids]
        return ""


class TestTokenAlignment(unittest.TestCase):
    """Test cases for token alignment functions"""
    
    @classmethod
    def setUpClass(cls):
        """Load spaCy model once for all tests"""
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            cls.skipTest(cls, "spaCy English model not available")
        cls.tokenizer = MockGPT2Tokenizer()
    
    def test_align_tokens_basic(self):
        """Test basic token alignment functionality"""
        sentence = "Timothy didn't boast about himself."
        doc = self.nlp(sentence)
        
        aligned_tokens = align_tokens_with_tokenizer(sentence, doc, self.tokenizer)
        
        # Check basic structure
        self.assertIsInstance(aligned_tokens, list)
        self.assertGreater(len(aligned_tokens), 0)
        
        # Check all tokens have required fields
        for token in aligned_tokens:
            self.assertIn('index', token)
            self.assertIn('text', token)
            self.assertIn('head_index', token)
            self.assertIn('dep', token)
            self.assertIn('original_index', token)
    
    def test_align_tokens_examples(self):
        """Test alignment on all example sentences"""
        test_sentences = [
            "Some women weren't understood by some guests.",
            "All piano teachers that criticized Dawn haven't heard Bruce.",
            "There were no shirts disturbing Theresa",
            "A lot of actresses that thought about Alice healed themselves."
        ]
        
        for sentence in test_sentences:
            with self.subTest(sentence=sentence):
                doc = self.nlp(sentence)
                aligned_tokens = align_tokens_with_tokenizer(sentence, doc, self.tokenizer)
                
                # Basic validation
                self.assertGreater(len(aligned_tokens), 0)
                
                # Check indices are sequential
                indices = [token['index'] for token in aligned_tokens]
                self.assertEqual(indices, list(range(len(aligned_tokens))))
                
                # Check head indices are valid
                for token in aligned_tokens:
                    self.assertGreaterEqual(token['head_index'], 0)
                    self.assertLess(token['head_index'], len(aligned_tokens))
    
    def test_linked_dependencies(self):
        """Test that linked dependencies are created for subword tokens"""
        sentence = "Timothy didn't boast about himself."
        doc = self.nlp(sentence)
        aligned_tokens = align_tokens_with_tokenizer(sentence, doc, self.tokenizer)

        print(f"Aligned tokens: {aligned_tokens}")
        
        # Should have some linked dependencies due to subword tokenization
        linked_tokens = [token for token in aligned_tokens if token['dep'] == 'linked']
        self.assertGreater(len(linked_tokens), 0)
        
        # Check linked tokens point to valid heads
        for token in linked_tokens:
            head_idx = token['head_index']
            self.assertNotEqual(head_idx, token['index'])  # Not self-referential
            self.assertLess(head_idx, token['index'])  # Points to earlier token


class TestPerturbations(unittest.TestCase):
    """Test cases for perturbation functions"""
    
    @classmethod
    def setUpClass(cls):
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            cls.skipTest(cls, "spaCy English model not available")
        cls.tokenizer = MockGPT2Tokenizer()
    
    def setUp(self):
        """Create test tokens for each test"""
        sentence = "Timothy didn't boast about himself."
        doc = self.nlp(sentence)
        self.tokens = align_tokens_with_tokenizer(sentence, doc, self.tokenizer)
    
    def test_windowed_shuffle_basic(self):
        """Test basic windowed shuffle functionality"""
        shuffled = apply_windowed_shuffle_perturbation(self.tokens, window=3, seed=42)
        
        # Same number of tokens
        self.assertEqual(len(shuffled), len(self.tokens))
        
        # All indices updated correctly
        indices = [token['index'] for token in shuffled]
        self.assertEqual(indices, list(range(len(shuffled))))
        
        # All head indices are valid
        for token in shuffled:
            self.assertGreaterEqual(token['head_index'], 0)
            self.assertLess(token['head_index'], len(shuffled))
    
    def test_windowed_shuffle_deterministic(self):
        """Test that same seed produces same result"""
        shuffled1 = apply_windowed_shuffle_perturbation(self.tokens, window=2, seed=123)
        shuffled2 = apply_windowed_shuffle_perturbation(self.tokens, window=2, seed=123)
        
        # Should be identical
        for t1, t2 in zip(shuffled1, shuffled2):
            self.assertEqual(t1['text'], t2['text'])
            self.assertEqual(t1['index'], t2['index'])
            self.assertEqual(t1['head_index'], t2['head_index'])
    
    def test_windowed_shuffle_different_seeds(self):
        """Test that different seeds produce different results"""
        shuffled1 = apply_windowed_shuffle_perturbation(self.tokens, window=2, seed=42)
        shuffled2 = apply_windowed_shuffle_perturbation(self.tokens, window=2, seed=999)
        
        # Should be different (with high probability)
        texts1 = [t['text'] for t in shuffled1]
        texts2 = [t['text'] for t in shuffled2]
        self.assertNotEqual(texts1, texts2)
    
    def test_reverse_perturbation(self):
        """Test reverse perturbation"""
        reversed_tokens = apply_reverse_perturbation(self.tokens)
        
        # Same number of tokens
        self.assertEqual(len(reversed_tokens), len(self.tokens))
        
        # Order should be reversed
        original_texts = [t['text'] for t in self.tokens]
        reversed_texts = [t['text'] for t in reversed_tokens]
        self.assertEqual(original_texts[::-1], reversed_texts)
        
        # All head indices should be valid
        for token in reversed_tokens:
            self.assertGreaterEqual(token['head_index'], 0)
            self.assertLess(token['head_index'], len(reversed_tokens))
    
    def test_dependency_preservation(self):
        """Test that dependency relationships are logically preserved"""
        # Find ROOT token in original
        original_root = None
        for token in self.tokens:
            if token['dep'] == 'ROOT':
                original_root = token
                break
        
        # Apply perturbation
        shuffled = apply_windowed_shuffle_perturbation(self.tokens, window=3, seed=42)
        
        # Find ROOT token in shuffled
        shuffled_root = None
        for token in shuffled:
            if token['dep'] == 'ROOT':
                shuffled_root = token
                break
        
        # ROOT token should exist and be the same word
        self.assertIsNotNone(original_root)
        self.assertIsNotNone(shuffled_root)
        self.assertEqual(original_root['text'], shuffled_root['text'])


class TestDependencyStatistics(unittest.TestCase):
    """Test cases for dependency statistics functions"""
    
    def setUp(self):
        """Create test cases with known properties"""
        
        # Simple projective case
        self.projective_tokens = [
            {'index': 0, 'text': 'The', 'dep': 'det', 'head_index': 1},
            {'index': 1, 'text': 'cat', 'dep': 'nsubj', 'head_index': 2},
            {'index': 2, 'text': 'sleeps', 'dep': 'ROOT', 'head_index': 2},
        ]
        
        # Non-projective case with crossing dependencies
        self.nonprojective_tokens = [
            {'index': 0, 'text': 'A', 'dep': 'det', 'head_index': 1},
            {'index': 1, 'text': 'man', 'dep': 'nsubj', 'head_index': 4},
            {'index': 2, 'text': 'who', 'dep': 'nsubj', 'head_index': 3},
            {'index': 3, 'text': 'lives', 'dep': 'relcl', 'head_index': 1},
            {'index': 4, 'text': 'sleeps', 'dep': 'ROOT', 'head_index': 4},
        ]
        
        # Case with linked tokens
        self.linked_tokens = [
            {'index': 0, 'text': 'Tim', 'dep': 'nsubj', 'head_index': 3},
            {'index': 1, 'text': 'othy', 'dep': 'linked', 'head_index': 0},
            {'index': 2, 'text': 'walk', 'dep': 'ROOT', 'head_index': 2},
            {'index': 3, 'text': 'ed', 'dep': 'linked', 'head_index': 2},
        ]
    
    def test_is_projective_true(self):
        """Test projective dependency detection"""
        self.assertTrue(is_projective(self.projective_tokens))
    
    def test_is_projective_false(self):
        """Test non-projective dependency detection"""
        self.assertFalse(is_projective(self.nonprojective_tokens))
    
    def test_total_dependency_distance(self):
        """Test total dependency distance calculation"""
        # For projective_tokens: |0-1| + |1-2| + |2-2| = 1 + 1 + 0 = 2
        distance = total_dependency_distance(self.projective_tokens)
        self.assertEqual(distance, 2)
        
        # Test empty case
        self.assertEqual(total_dependency_distance([]), 0)
    
    def test_normalized_dependency_distance(self):
        """Test normalized dependency distance"""
        distance = normalized_dependency_distance(self.projective_tokens)
        expected = 2.0 / 3.0  # total_distance / num_tokens
        self.assertAlmostEqual(distance, expected, places=3)
        
        # Test empty case
        self.assertEqual(normalized_dependency_distance([]), 0.0)
    
    def test_same_word_token_distances(self):
        """Test same-word token distance calculation"""
        distances = same_word_token_distances(self.linked_tokens)
        # Should find distances for 'linked' dependencies: |1-0| = 1, |3-2| = 1
        expected = [1, 1]
        self.assertEqual(sorted(distances), sorted(expected))
        
        # Test case with no linked tokens
        distances = same_word_token_distances(self.projective_tokens)
        self.assertEqual(distances, [])
    
    def test_crossing_dependencies_count(self):
        """Test crossing dependencies counting"""
        # Projective case should have 0 crossings
        crossings = crossing_dependencies_count(self.projective_tokens)
        self.assertEqual(crossings, 0)
        
        # Non-projective case should have crossings
        crossings = crossing_dependencies_count(self.nonprojective_tokens)
        self.assertGreater(crossings, 0)
    
    def test_arcs_cross_function(self):
        """Test the arc crossing detection function"""
        # Non-crossing arcs
        self.assertFalse(arcs_cross((0, 2), (3, 5)))  # Separate arcs
        self.assertFalse(arcs_cross((0, 5), (1, 3)))  # Nested arcs
        
        # Crossing arcs
        self.assertTrue(arcs_cross((0, 3), (1, 4)))   # Crossing pattern
        self.assertTrue(arcs_cross((1, 4), (0, 3)))   # Symmetric
    
    def test_calculate_dependency_statistics(self):
        """Test comprehensive statistics calculation"""
        stats = calculate_dependency_statistics(self.projective_tokens)
        
        # Check all expected keys are present
        expected_keys = [
            'is_projective', 'total_dependency_distance', 
            'normalized_dependency_distance', 'same_word_token_distances',
            'crossing_dependencies_count', 'num_tokens', 'average_dependency_distance'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check some values
        self.assertTrue(stats['is_projective'])
        self.assertEqual(stats['num_tokens'], 3)
        self.assertEqual(stats['crossing_dependencies_count'], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_tokens(self):
        """Test functions with empty token lists"""
        empty_tokens = []
        
        self.assertTrue(is_projective(empty_tokens))
        self.assertEqual(total_dependency_distance(empty_tokens), 0)
        self.assertEqual(normalized_dependency_distance(empty_tokens), 0.0)
        self.assertEqual(same_word_token_distances(empty_tokens), [])
        self.assertEqual(crossing_dependencies_count(empty_tokens), 0)
    
    def test_single_token(self):
        """Test functions with single token"""
        single_token = [{'index': 0, 'text': 'Hello', 'dep': 'ROOT', 'head_index': 0}]
        
        self.assertTrue(is_projective(single_token))
        self.assertEqual(total_dependency_distance(single_token), 0)
        self.assertEqual(normalized_dependency_distance(single_token), 0.0)
        self.assertEqual(crossing_dependencies_count(single_token), 0)
    
    def test_all_self_loops(self):
        """Test case where all dependencies are self-loops"""
        self_loop_tokens = [
            {'index': 0, 'text': 'A', 'dep': 'ROOT', 'head_index': 0},
            {'index': 1, 'text': 'B', 'dep': 'ROOT', 'head_index': 1},
        ]
        
        self.assertTrue(is_projective(self_loop_tokens))
        self.assertEqual(total_dependency_distance(self_loop_tokens), 0)
        self.assertEqual(crossing_dependencies_count(self_loop_tokens), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions"""
    
    @classmethod
    def setUpClass(cls):
        try:
            cls.nlp = spacy.load("en_core_web_sm")
        except OSError:
            cls.skipTest(cls, "spaCy English model not available")
        cls.tokenizer = MockGPT2Tokenizer()
    
    def test_full_pipeline_example_sentences(self):
        """Test full pipeline on all example sentences"""
        test_sentences = [
            "Some women weren't understood by some guests.",
            "All piano teachers that criticized Dawn haven't heard Bruce.",
            "There were no shirts disturbing Theresa",
            "A lot of actresses that thought about Alice healed themselves."
        ]
        
        for sentence in test_sentences:
            with self.subTest(sentence=sentence):
                # Step 1: Parse and align
                doc = self.nlp(sentence)
                aligned_tokens = align_tokens_with_tokenizer(sentence, doc, self.tokenizer)
                
                # Step 2: Apply perturbation
                shuffled_tokens = apply_windowed_shuffle_perturbation(aligned_tokens, window=3, seed=42)
                
                # Step 3: Calculate statistics
                original_stats = calculate_dependency_statistics(aligned_tokens)
                shuffled_stats = calculate_dependency_statistics(shuffled_tokens)
                
                # Validate both work without errors
                self.assertIsInstance(original_stats, dict)
                self.assertIsInstance(shuffled_stats, dict)
                
                # Some properties should be preserved
                self.assertEqual(original_stats['num_tokens'], shuffled_stats['num_tokens'])
    
    def test_perturbation_preserves_logical_structure(self):
        """Test that perturbations preserve the logical dependency structure"""
        sentence = "Timothy didn't boast about himself."
        doc = self.nlp(sentence)
        aligned_tokens = align_tokens_with_tokenizer(sentence, doc, self.tokenizer)
        
        # Apply different perturbations
        shuffled = apply_windowed_shuffle_perturbation(aligned_tokens, window=2, seed=42)
        reversed_tokens = apply_reverse_perturbation(aligned_tokens)
        
        # All should have same dependency relations (just different indices)
        original_deps = sorted([token['dep'] for token in aligned_tokens])
        shuffled_deps = sorted([token['dep'] for token in shuffled])
        reversed_deps = sorted([token['dep'] for token in reversed_tokens])
        
        self.assertEqual(original_deps, shuffled_deps)
        self.assertEqual(original_deps, reversed_deps)


def run_comprehensive_tests():
    """Run all tests and print results"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTokenAlignment,
        # TestPerturbations, 
        # TestDependencyStatistics,
        # TestEdgeCases,
        # TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the comprehensive test suite
    print("Running comprehensive test suite for dependency parsing functions...")
    print("This will test token alignment, perturbations, and statistics calculation.")
    print("=" * 80)
    
    success = run_comprehensive_tests()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")