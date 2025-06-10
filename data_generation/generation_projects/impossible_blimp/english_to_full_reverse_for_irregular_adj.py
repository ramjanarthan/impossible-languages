
from data_generation.utils import data_generator
from data_generation.utils.constituent_building import *
from data_generation.utils.conjugate import *
from data_generation.utils.randomize import choice
from functools import reduce
from data_generation.utils.vocab_sets import *
from data_generation.utils.impossible_utils import PERTURBATIONS

class EnglishToFullReverseForIrregularAdjGenerator(data_generator.ParallelBenchmarkGenerator):
    def __init__(self):
        super().__init__(field="morphology",
                         linguistics="irregular_forms",
                         uid="english_to_full_reverse_for_irregular_adj",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)
        self.all_trans_en_verbs = get_all("special_en_form", "1", all_transitive_verbs)

    def sample(self):
        # The eaten pie was delicious
        # THE V_en  N1  cop adj
        # The ate    pie was delicious
        # THE V_past N1 cop adj

        V_base = choice(self.all_trans_en_verbs)
        while (' ' in V_base[0]):
            V_base = choice(self.all_trans_en_verbs)
        Verbs = get_all("root", V_base["root"])
        V_past = get_all("past", "1", Verbs)
        V_en = get_all("en", "1", Verbs)
        N1 = choice(get_matches_of(V_base, "arg_2", all_common_nouns))
        cop = return_copula(N1)
        adj = choice(get_matched_by(N1, "arg_1", all_adjectives))

        data = {
            "dataset_A_grammatical": "The %s %s %s %s." % (V_en[0][0], N1[0], cop[0], adj[0]),
            "dataset_A_ungrammatical": "The %s %s %s %s." % (V_past[0][0], N1[0], cop[0], adj[0]),
            "two_prefix_prefix_good": "The %s" % (V_en[0][0]),
            "two_prefix_prefix_bad": "The %s" % (V_past[0][0]),
            "two_prefix_word": N1[0]
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


binding_generator = EnglishToFullReverseForIrregularAdjGenerator()
binding_generator.generate_paradigm(number_to_generate=500, rel_output_path="outputs/impossible_blimp/%s.jsonl" % binding_generator.uid)