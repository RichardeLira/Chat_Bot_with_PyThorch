# Library imports

import nltk 
nltk.download("punkt") 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# nltk.download('stopwords')
import re
import string
import numpy as np 

# ------------ Pre Processing ---------------- #


def tokenize(sentence): 
    return nltk.word_tokenize(sentence)


def stem(word): 
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words): 
    token_setence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in token_setence:
            bag[idx] = 1.0

    return bag 



def clean_text(instance):
    instance = instance.lower()
    instance = re.sub('\[.*?\]', ' ', instance)
    instance = re.sub('https?://\S+|www\.\S+', ' ', instance)
    instance = re.sub('<.*?>+', ' ', instance)
    instance = re.sub('[%s]' % re.escape(string.punctuation), ' ', instance)
    instance = re.sub('\n', '', instance)
    instance = re.sub('\w*\d\w*', ' ', instance)
    return instance

def remove_extra_spaces(instance):
    instance = re.sub(' +', ' ', instance)
    return instance

def RemoveStopWords(instance):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    palavras = [i for i in instance.split() if not i in stopwords]
    return (" ".join(palavras))

