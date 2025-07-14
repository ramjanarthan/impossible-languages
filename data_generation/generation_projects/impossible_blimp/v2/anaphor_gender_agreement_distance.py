from data_generation.utils import data_generator
from data_generation.utils.constituent_building import *
from data_generation.utils.conjugate import *
from data_generation.utils.randomize import choice
from functools import reduce
from data_generation.utils.vocab_sets import *
import datetime

class AnaphorGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="morphology",
            linguistics="anaphor_agreement",
            uid="anaphor_gender_agreement_distance",
            simple_lm_method=True,
            one_prefix_method=True,
            two_prefix_method=False,
            lexically_identical=False
        )
        self.all_safe_nouns = np.setdiff1d(all_singular_nouns, all_singular_neuter_animate_nouns)
        self.all_singular_reflexives = reduce(np.union1d, (get_all("expression", "himself"),
                                                           get_all("expression", "herself"),
                                                           get_all("expression", "itself")))
    
    def make_metadata_dict(self):
        metadata = {
            "UID": self.uid,
        }
        return metadata

    def sample(self):
        V1 = choice(all_refl_preds)
        # Add an adjective and a PP modifier to the subject NP
        base_noun = choice(get_matches_of(V1, "arg_1", get_matches_of(V1, "arg_2", self.all_safe_nouns)))
        adj = choice(get_matched_by(base_noun, "arg_1", all_adjectives))
        # Pick a simple, inanimate object as PP attachment to avoid introducing animacy or agreement confounds
        pp_obj = choice(get_all("animate", "0", get_all("sg", "1", all_common_nouns)))
        prepositions = get_all("category", "(S[pred]\\NP)/NP")
        prep_entry = choice(prepositions)
        prep = prep_entry[0]  # surface form/expression
        # Choose determiner: 'the', or 'a'/'an' based on adjective
        vowels = set("aeiouAEIOU")
        use_indef = choice([True, False])
        if use_indef and base_noun["sg"] == "1":
            det = "an" if adj[0][0] in vowels else "a"
        else:
            det = "the"
        # Build the full subject NP
        N1_str = f"{det} {adj[0]} {base_noun[0]} {prep} the {pp_obj[0]}"
        N1 = (N1_str, base_noun[1])  # preserve index for compatibility
        refl_match = choice(get_matched_by(base_noun, "arg_1", all_reflexives))
        refl_mismatch = choice(np.setdiff1d(self.all_singular_reflexives, [refl_match]))

        V1 = conjugate(V1, base_noun)

        data = {
            "sentence_good": "%s %s %s." % (N1[0], V1[0], refl_match[0]),
            "sentence_bad": "%s %s %s." % (N1[0], V1[0], refl_mismatch[0]),
        }
        return data, data["sentence_good"]

generator = AnaphorGenerator()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
generator.generate_paradigm(number_to_generate=1000, rel_output_path="outputs/impossible_blimp/v2/%s_%s.jsonl" % (generator.uid, timestamp))













