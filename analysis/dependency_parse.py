import spacy
from spacy import displacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Example sentence to parse
sentence = "Timothy didn't boast about himself."
reverse_sentence = ". himself about🅁 boast't didnothyTim"
reverse_parial_sentence = "Timothy didn't boast🅁. himself about"
shuffle_det_sentence = "aboutTim boastothy himself't. didn"
shuffle_non_det_sentence = ". himself didn about boastTimothy't"

# Parse the sentence using spaCy
doc = nlp(sentence)

# Visualize the dependency tree with default settings
displacy.serve(doc, style='dep', port=5004)