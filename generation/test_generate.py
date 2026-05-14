from data_generation.utils.impossible_utils import (
    PERTURBATION_TO_HF_MODEL_NAME,
    VALID_UNDO_PERTURBATION_KEYS,
    UNDO_PERTURBATIONS,
    VALID_PERTURBATION_KEYS,
    PERTURBATIONS,
)
import unittest

test_cases = [
    "A short one",
    "Small boy",
    "Hi.",
    "The quick brown fox jumps over the lazy dog",
    "He's very happy today!",
    "New technologies are transforming the world.",
    "Hi there!",
    "Won't you join us for dinner tonight?",
    "Data science is an interdisciplinary field that is growing rapidly amongst professionals all over the world, especially in places where Pokemon is wdiely played.",
]


class UndoPerturbationTestResults(unittest.TextTestResult):
    def addSubTest(self, test, subtest, outcome):
        # handle failures calling base class
        super(UndoPerturbationTestResults, self).addSubTest(test, subtest, outcome)
        # add to total number of tests run
        self.testsRun += 1


class TestUndoPerturbations(unittest.TestCase):
    def test_undo_perturbations(self):
        for test_sentence in test_cases:
            for key in VALID_UNDO_PERTURBATION_KEYS:
                with self.subTest(key=key, test_sentence=test_sentence):
                    output = PERTURBATIONS[key]["perturbation_function"](test_sentence)

                    # Apply the undo perturbation function
                    corrected_output = UNDO_PERTURBATIONS[key]["perturbation_function"](
                        output
                    )
                    corrected_sentence = UNDO_PERTURBATIONS[key][
                        "gpt2_tokenizer"
                    ].decode(corrected_output, skip_special_tokens=True)

                    self.assertEqual(test_sentence, corrected_sentence)


if __name__ == "__main__":
    unittest.main(
        testRunner=unittest.TextTestRunner(resultclass=UndoPerturbationTestResults)
    )
