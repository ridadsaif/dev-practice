# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:28:45 2023

@author: ridad
"""

import numpy as np
import re
from nltk.corpus import stopwords
import math
import operator
import scipy
from nltk.stem import PorterStemmer
import string
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string
import pandas as pd

with open("model1.pickle", "rb") as handle1:
    model1 = pickle.load(handle1)
with open("model2.pickle", "rb") as handle2:
    model2 = pickle.load(handle2)
model = {**model1, **model2}
en_words = list(model.keys())

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def extract_nouns(text):
    # Tokenize the input text into words
    words = word_tokenize(text)
    # Perform POS tagging
    tagged_words = pos_tag(words)
    # Extract nouns from the tagged words
    nouns = [word for word, tag in tagged_words if tag.startswith('N')]
    return nouns

def preprocess_word(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    cleaned_word = letters_only_text.lower().split()

    if cleaned_word:
        return cleaned_word[0]
    else:
        return ''

def preprocess_text(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = list(set(stopwords.words("english")))
    stopword_set.append("among")
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    return cleaned_words

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0  # To avoid division by zero for zero vectors
    
    similarity = dot_product / (magnitude_v1 * magnitude_v2)
    return similarity

def cosine_distance_wordembedding_method(word, text):
    if word in en_words:
        vector_one = model[word]
    else:
        vector_one = []
    vector_two = np.mean(
        [model[word] for word in preprocess_text(text) if word in en_words], axis=0
    )

    cosine = cosine_similarity(vector_one, vector_two)

    return cosine

def remove_punctuation(input_string):
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    # Use the translate method to remove punctuation
    return input_string.translate(translator)



def extract_keywords(df_text, groups):
    all_keywords = []
    for index, row in df_text.iterrows():
        group_name = row['group name']
        text = row['Google Translate']
        nouns_list = extract_nouns(text)
        group_keywords = groups[group_name]
        scores = {}
        for word in nouns_list:
            word = remove_punctuation(word)
            if word.lower() in en_words:
                if '-' in word:
                    preprocessed_word = word
                else:
                    preprocessed_word = preprocess_word(word)
                if preprocessed_word != '':
                    score = cosine_distance_wordembedding_method(preprocessed_word, group_keywords)
                    scores[word] = score
        if scores:
            selected_words = sorted(scores, key=scores.get, reverse=True)[:3]
            keywords = ' '.join(selected_words)
            all_keywords.append(keywords)
        else:
            all_keywords.append('No keyword found')
    
    return all_keywords
        

if __name__ == "__main__":
    df_text = pd.read_csv('Prototype Report for bKash - Raw Data.csv')
    
    groups = {'prabashibangladesh': 'expatriate Bangladesh immigrant non-resident',
     'thefreelancersofbd': 'freelancers information technomogly professionalism job',
     'eeCAB': 'e-commerce',
     'BrandPractitionersBD': 'brand marketing',
     'Travellers of Bangladesh': 'travel vacation adventure nature'}
    
    keywords_list = extract_keywords(df_text, groups)
    
    df_text['Relevant Keywords'] = keywords_list
    
    df_text.to_csv("Dataset_with_relevant_keywords.csv")