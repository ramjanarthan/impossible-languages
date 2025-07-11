import spacy
from spacy import displacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Example sentence to parse
sentence = "Apple's CEO Tim Cook visited the company's headquarters in Cupertino."

# Parse the sentence using spaCy
doc = nlp(sentence)

# Visualize the dependency tree with default settings
displacy.serve(doc, style='dep', port=5001)