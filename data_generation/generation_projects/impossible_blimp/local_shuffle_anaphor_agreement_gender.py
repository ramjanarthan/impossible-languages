
from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from functools import reduce
from utils.vocab_sets import *
from numpy.random import default_rng

class ImpossibleLocalShuffleAnaphorGenerator(data_generator.ImpossibleBenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="morphology",
            linguistics="anaphor_agreement",
            uid="impossible_anaphor_gender_agreement",
            simple_lm_method=True,
            one_prefix_method=True,
            two_prefix_method=False,
            lexically_identical=False
        )
        self.all_safe_nouns = np.setdiff1d(all_singular_nouns, all_singular_neuter_animate_nouns)
        self.all_singular_reflexives = reduce(np.union1d, (get_all("expression", "himself"),
                                                           get_all("expression", "herself"),
                                                           get_all("expression", "itself")))
        self.seed = 42

    def sample(self):
        # John knows himself
        # N1   V1    refl_match
        # John knows itself
        # N1   V1    refl_mismatch

        # Impossible sentences
        # knows himself John
        # V1   refl_match   N1
        # knows itself John
        # V1   refl_mismatch   N1

        V1 = choice(all_refl_preds)
        N1 = N_to_DP_mutate(choice(get_matches_of(V1, "arg_1", get_matches_of(V1, "arg_2", self.all_safe_nouns))))
        refl_match = choice(get_matched_by(N1, "arg_1", all_reflexives))
        refl_mismatch = choice(np.setdiff1d(self.all_singular_reflexives, [refl_match]))

        V1 = conjugate(V1, N1)

        data = {
            "sentence_good": "%s %s %s." % (N1[0], V1[0], refl_match[0]),
            "sentence_bad": "%s %s %s." % (N1[0], V1[0], refl_mismatch[0]),
            "one_prefix_prefix": "%s %s" % (N1[0], V1[0]),
            "one_prefix_word_good": refl_match[0],
            "one_prefix_word_bad": refl_mismatch[0],
        }

        # Impossible sentences
        impossible_data = {
            "impossible_sentence_good": self.__perturb_shuffle_local(data["sentence_good"], self.seed),  
            "impossible_sentence_bad": self.__perturb_shuffle_local(data["sentence_bad"], self.seed),
        }

        # merge data and impossible_data
        data = {**data, **impossible_data}

        return data, data["sentence_good"]
    
    def __perturb_shuffle_local(self, sent, seed, window=3):
        # Get sentence text 
        tokens = sent.split(" ")

        # Shuffle tokens in batches of size window
        shuffled_tokens = []
        for i in range(0, len(tokens), window):
            batch = tokens[i:i+window].copy()
            default_rng(seed).shuffle(batch)
            shuffled_tokens += batch

        return " ".join(shuffled_tokens)


binding_generator = ImpossibleLocalShuffleAnaphorGenerator()
binding_generator.generate_paradigm(number_to_generate=1000, rel_output_path="outputs/impossible_blimp/%s.jsonl" % binding_generator.uid)