# Library imports

import nltk 
nltk.download("punkt") 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# ------------ Pre Processing ---------------- #


def tokenize(sentence): 
    return nltk.word_tokenize(sentence)


def stem(word): 
    return stemmer.stem(word.lower())