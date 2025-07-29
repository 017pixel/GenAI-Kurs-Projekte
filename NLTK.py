import nltk

print('Downloading NLTK resources...')

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize

time.sleep(3)  # Simulate a delay for downloading resources

print("-----------------------------------------------------------------------------------------------------------------------")
import time

sample_text = 'Ich lerne Python und NLTK. Es ist eine tolle Bibliothek für die Verarbeitung natürlicher Sprache.'
tokens = nltk.word_tokenize(sample_text.lower())

print('Tokens:', tokens)

print("-----------------------------------------------------------------------------------------------------------------------")