
from data_generation.utils import data_generator
from data_generation.utils.constituent_building import *
from data_generation.utils.conjugate import *
from data_generation.utils.randomize import choice
from functools import reduce
from data_generation.utils.vocab_sets import *
from data_generation.utils.impossible_utils import PERTURBATIONS

class EnglishToFullReverseForAnaphorAgreementGenderGenerator(data_generator.ParallelBenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="morphology",
            linguistics="anaphor_agreement",
            uid="english_to_full_reverse_for_anaphor_agreement_gender",
            simple_lm_method=True,
            one_prefix_method=False,
            two_prefix_method=True,
            lexically_identical=False
        )
        self.all_safe_nouns = np.setdiff1d(all_singular_nouns, all_singular_neuter_animate_nouns)
        self.all_singular_reflexives = reduce(np.union1d, (get_all("expression", "himself"),
                                                           get_all("expression", "herself"),
                                                           get_all("expression", "itself")))
        self.seed = 0

    def sample(self):
        # John knows himself
        # N1   V1    refl_match
        # John knows itself
        # N1   V1    refl_mismatch

        # Impossible sentences
        # himself knows John
        # V1   refl_match   N1
        # itself knows John
        # V1   refl_mismatch   N1

        V1 = choice(all_refl_preds)
        N1 = N_to_DP_mutate(choice(get_matches_of(V1, "arg_1", get_matches_of(V1, "arg_2", self.all_safe_nouns))))
        refl_match = choice(get_matched_by(N1, "arg_1", all_reflexives))
        refl_mismatch = choice(np.setdiff1d(self.all_singular_reflexives, [refl_match]))

        V1 = conjugate(V1, N1)

        data = {
            "dataset_A_grammatical": "%s %s %s." % (N1[0], V1[0], refl_match[0]),
            "dataset_A_ungrammatical": "%s %s %s." % (N1[0], V1[0], refl_mismatch[0]),
            "one_prefix_prefix": "%s %s" % (N1[0], V1[0]),
            "one_prefix_word_good": refl_match[0],
            "one_prefix_word_bad": refl_mismatch[0],
        }

        perturbation = PERTURBATIONS["reverse_full"]

        # Impossible sentences
        impossible_sentence_good = perturbation["perturbation_function"](data["dataset_A_grammatical"])
        impossible_sentence_bad = perturbation["perturbation_function"](data["dataset_A_ungrammatical"])

        impossible_sentence_good = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), impossible_sentence_good))
        impossible_sentence_bad = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), impossible_sentence_bad))

        data["dataset_B_grammatical"] = impossible_sentence_good
        data["dataset_B_ungrammatical"] = impossible_sentence_bad

        return data, data["dataset_A_grammatical"]


binding_generator = EnglishToFullReverseForAnaphorAgreementGenderGenerator()
binding_generator.generate_paradigm(number_to_generate=1000, rel_output_path="outputs/impossible_blimp/%s.jsonl" % binding_generator.uid)