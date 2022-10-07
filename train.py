import json 
from Nltk_utils import tokenize, stem, bag_of_words, clean_text,remove_extra_spaces, RemoveStopWords
import numpy as np 

with open("intents.json", 'r') as f:
    intents = json.load(f)

# Bulding the data traning

all_words = []
tags = []
xy = []

# Accessing the Json file e colect the data 

# Applyin preprocessing 

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

all_words = [clean_text(w) for w in all_words]
all_words = [stem(w) for w in all_words]
all_words = [remove_extra_spaces(w) for w in all_words]
all_words = [RemoveStopWords(w) for w in all_words]
all_words = sorted(set(all_words))
tags      = sorted(set(tags))

# Separate the data in Train

print(tags)

X_train = []
y_train = []


for(pattern_setence, tag) in xy:
    bag = bag_of_words(pattern_setence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #1 Hot encode vector 


X_train = np.array(X_train)
y_train = np.array(y_train)