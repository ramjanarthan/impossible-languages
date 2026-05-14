color_pal = {
    "english": "#FFC0CB",
    "reverse_full": "#F27E9D",
    "reverse_partial": "#592A37",
    "shuffle_local3": "#77D9CF",
    "shuffle_local5": "#5AB8C6",
    "shuffle_even_odd": "#85AFEB",
    "shuffle_local10": "#05A6A6",
    "shuffle_deterministic21": "#144045",
    "shuffle_nondeterministic": "#666666",
}


def initialize_style():
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "Futura"
    return color_pal
