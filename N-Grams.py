import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

import time

text = "Ich lerne Python und NLTK. Es ist eine Python Bibliothek für die Verarbeitung natürlicher Sprache."
tokens = word_tokenize(text)

#Unigram
unigrams_list = list(ngrams(tokens, 1))
print("Unigrams:", unigrams_list)
print("")

#Bigram
bigrams_list = list(ngrams(tokens, 2))
print("Bigrams:", bigrams_list)
print("")

#Trigram
trigrams_list = list(ngrams(tokens, 3))
print("Trigrams:", trigrams_list)
print("")