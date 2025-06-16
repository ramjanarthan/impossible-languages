
from data_generation.utils import data_generator
from data_generation.utils.constituent_building import *
from data_generation.utils.conjugate import *
from data_generation.utils.randomize import choice
from functools import reduce
from data_generation.utils.vocab_sets import *
from data_generation.utils.impossible_utils import PERTURBATIONS

class EnglishToPartialReverseForEllipsisGenerator(data_generator.ParallelBenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                            linguistics="ellipsis",
                            uid="english_to_partial_reverse_for_ellipsis",
                            simple_lm_method=True,
                            one_prefix_method=False,
                            two_prefix_method=False,
                            lexically_identical=False)
        self.safe_verbs = all_transitive_verbs
        self.plural_dets = [("two", "three"),
                            ("three", "two"),
                            ("four", "five"),
                            ("five", "four"),
                            ("several", "a few"),
                            ("a few", "several"),
                            ("many", "few"),
                            ("few", "many"),
                            ("a few", "a lot"),
                            ("a lot of", "a few"),
                            ("no", "some"),
                            ("three", "more"),
                            ("three", "fewer"),
                            ("three", "at least as many"),
                            ("three", "almost as many"),
                            ]

        self.singular_dets = [("one", "three"),
                            ("one", "two"),
                            ("one", "five"),
                            ("one", "four"),
                            ("one", "a few"),
                            ("one", "several"),
                            ("one", "few"),
                            ("one", "many"),
                            ("one", "a lot"),
                            ("one", "some"),
                            ("one", "more"),
                            ("one", "at least as many")
                            ]
        self.safe_objs = np.setdiff1d(all_nominals, all_proper_names)

    def sample(self):
        # John  has  had two green cups and Jane  has had three.
        # Subj1 Aux1 V   D1  Adj   Obj  AND Subj2 Aux2 V   D2
        # John  has  had two cups and Jane  has  had three green.
        # Subj1 Aux1 V   D1  Obj  AND Subj2 Aux2 V   D2    Adj

        # Impossible sentences
        # John  has  had two green 🅁 three had has Jane and cups
        # Subj1 Aux1 V   D1  Adj   Obj  AND Subj2 Aux2 V   D2
        # John  has  had two cups 🅁 green three had has Jane and.
        # Subj1 Aux1 V   D1  Obj  AND Subj2 Aux2 V   D2    Adj

        V = choice(self.safe_verbs)
        Subj1 = choice(get_matches_of(V, "arg_1", all_nominals))
        Subj2 = choice(get_matches_of(V, "arg_1", all_nominals), avoid=Subj1)
        Subj1 = N_to_DP_mutate(Subj1)
        Subj2 = N_to_DP_mutate(Subj2)
        Aux1 = return_aux(V, subj=Subj1)
        if is_match_disj(Subj2, Aux1["arg_1"]):
            Aux2 = Aux1
        else:
            Aux2 = return_aux(V, subj=Subj2)
        Obj = choice(get_matches_of(V, "arg_2", get_all("mass", "0", self.safe_objs)))
        Adj = choice(get_matched_by(Obj, "arg_1", all_adjectives))
        if Obj["pl"] == "0":
            (D1, D2) = random.choice(self.singular_dets)
        else:
            (D1, D2) = random.choice(self.plural_dets)

        data = {
            "dataset_A_grammatical": "%s %s %s %s %s %s and %s %s %s %s." % (Subj1[0], Aux1[0], V[0], D1, Adj[0], Obj[0], Subj2[0], Aux2[0], V[0], D2),
            "dataset_A_ungrammatical": "%s %s %s %s %s and %s %s %s %s %s." % (Subj1[0], Aux1[0], V[0], D1, Obj[0], Subj2[0], Aux2[0], V[0], D2, Adj[0]),
        }

        perturbation = PERTURBATIONS["reverse_partial"]

        # Impossible sentences
        impossible_sentence_good = perturbation["perturbation_function"](data["dataset_A_grammatical"])
        impossible_sentence_bad = perturbation["perturbation_function"](data["dataset_A_ungrammatical"])

        impossible_sentence_good = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), impossible_sentence_good))
        impossible_sentence_bad = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), impossible_sentence_bad))

        data["dataset_B_grammatical"] = impossible_sentence_good
        data["dataset_B_ungrammatical"] = impossible_sentence_bad

        return data, data["dataset_A_grammatical"]


binding_generator = EnglishToPartialReverseForEllipsisGenerator()
binding_generator.generate_paradigm(number_to_generate=1000, rel_output_path="outputs/impossible_blimp/%s.jsonl" % binding_generator.uid)