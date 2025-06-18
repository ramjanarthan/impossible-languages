from data_generation.utils import data_generator
from data_generation.utils.constituent_building import *
from data_generation.utils.conjugate import *
from data_generation.utils.randomize import choice
from data_generation.utils.vocab_sets import *
import datetime

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="morphology",
                         linguistics="irregular_forms",
                         uid="irregular_past_participle_adjectives",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)
        self.all_trans_en_verbs = get_all("special_en_form", "1", all_transitive_verbs)

    def make_metadata_dict(self):
        metadata = {
            "UID": self.uid,
        }
        return metadata

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
            "sentence_good": "The %s %s %s %s." % (V_en[0][0], N1[0], cop[0], adj[0]),
            "sentence_bad": "The %s %s %s %s." % (V_past[0][0], N1[0], cop[0], adj[0]),
        }
        return data, data["sentence_good"]

generator = AgreementGenerator()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
generator.generate_paradigm(number_to_generate=1000, rel_output_path="outputs/impossible_blimp/v2/%s_%s.jsonl" % (generator.uid, timestamp))

