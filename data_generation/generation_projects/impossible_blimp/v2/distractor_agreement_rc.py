from data_generation.utils import data_generator
from data_generation.utils.constituent_building import *
from data_generation.utils.conjugate import *
from data_generation.utils.randomize import choice
from data_generation.utils.vocab_sets_dynamic import *
from data_generation.utils.impossible_utils import PERTURBATIONS

class ImpossibleDistractorAgreementRCGenerator(data_generator.BenchmarkGenerator):
    PERTURBATION_KEYS_FOR_DATA_GENERATION = ["shuffle_nondeterministic", "shuffle_deterministic21", "shuffle_local3", "shuffle_local5", "shuffle_even_odd", "reverse_control", "reverse_partial", "reverse_full"]
    PERTURBATION_KEYS_FOR_EVALUATION = ["english"] + PERTURBATION_KEYS_FOR_DATA_GENERATION

    def __init__(self):
        super().__init__(field="morphology",
                         linguistics="subject_verb_agreement",
                         uid="impossible_distractor_agreement_relative_clause",
                         simple_lm_method=True,
                         one_prefix_method=True,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.all_reg_nouns = get_all_conjunctive([("category", "N"), ("irrpl", "")], get_all_common_nouns())
        self.safe_emb_verbs = get_all_transitive_verbs()
        self.safe_mat_verbs = np.setdiff1d(get_all_verbs(), get_all("past", "1"))

    def sample(self):
        # The cat   that was      eating the mice is        sleeping
        #     Subj  Rel  Aux_Emb  V_emb  Obj_emb  Aux_agree V_mat_agree args
        # The cat   that was      eating the mice are           sleeping
        #     Subj  Rel  Aux_Emb  V_emb  Obj_emb  Aux_not_agree V_mat_not_agree args

        V_emb = None
        while V_emb is None:
            V_mat_agree = choice(self.safe_mat_verbs)
            subj = N_to_DP_mutate(choice(get_matches_of(V_mat_agree, "arg_1", self.all_reg_nouns)))
            rel = choice(get_matched_by(subj, "arg_1", get_all("category_2", "rel")))
            if V_mat_agree["finite"] == "1":
                if V_mat_agree["3sg"] == "1":
                    V_mat_not_agree = choice(get_all_conjunctive([("pres", "1"), ("3sg", "0")], get_all("root", V_mat_agree["root"])))
                else:
                    V_mat_not_agree = choice(get_all_conjunctive([("pres", "1"), ("3sg", "1")], get_all("root", V_mat_agree["root"])))
            else:
                V_mat_not_agree = V_mat_agree

            if subj["pl"] == "1":
                obj_emb = N_to_DP_mutate(choice(get_matches_of(V_mat_not_agree, "arg_1", get_all_singular_nouns())))
            else:
                obj_emb = N_to_DP_mutate(choice(get_matches_of(V_mat_not_agree, "arg_1", get_all_plural_nouns())))

            try:
                V_emb = choice(get_matched_by(subj, "arg_1", get_matched_by(obj_emb, "arg_2", self.safe_emb_verbs)))
            except IndexError:
                pass

        Aux_emb = return_aux(V_emb, subj)

        Auxs = require_aux_agree(V_mat_agree, subj)
        Aux_agree = Auxs["aux_agree"]
        Aux_not_agree = Auxs["aux_nonagree"]
        V_mat_args = verb_args_from_verb(V_mat_agree, subj=subj, aux=Aux_agree)

        data = {
            "sentence_good_english": "%s %s %s %s %s %s %s %s." % (subj[0], rel[0], Aux_emb[0], V_emb[0], obj_emb[0], Aux_agree, V_mat_agree[0], join_args(V_mat_args["args"])),
            "sentence_bad_english": "%s %s %s %s %s %s %s %s." % (subj[0], rel[0], Aux_emb[0], V_emb[0], obj_emb[0], Aux_not_agree, V_mat_not_agree[0], join_args(V_mat_args["args"])),
        }

        for perturbation_key in ImpossibleDistractorAgreementRCGenerator.PERTURBATION_KEYS_FOR_DATA_GENERATION:
            perturbation = PERTURBATIONS[perturbation_key]

            # Impossible sentences
            impossible_sentence_good = perturbation["perturbation_function"](data["sentence_good_english"])
            impossible_sentence_bad = perturbation["perturbation_function"](data["sentence_bad_english"])

            impossible_sentence_good = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), impossible_sentence_good))
            impossible_sentence_bad = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), impossible_sentence_bad))

            data[f"sentence_good_{perturbation_key}"] = impossible_sentence_good
            data[f"sentence_bad_{perturbation_key}"] = impossible_sentence_bad

        return data, data["sentence_good_english"]

generator = ImpossibleDistractorAgreementRCGenerator()
generator.generate_paradigm(number_to_generate=50, rel_output_path="outputs/impossible_blimp/v2/%s.jsonl" % generator.uid)
