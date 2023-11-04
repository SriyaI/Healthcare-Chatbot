import random
import string
import numpy as np
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize():
    file = open('corpus.txt', 'r')
    corpus = file.read()
    sentence_tokens = nltk.sent_tokenize(corpus)
    word_tokens = nltk.word_tokenize(corpus)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in word_tokens]
    return sentence_tokens, lemmatized_tokens

def respond(user_query):
    bot_response = ''
    sentence_tokens, lemmatized_tokens = tokenize_and_lemmatize()
    sentence_tokens.append(user_query)
    tfidf_obj = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_obj.fit_transform(sentence_tokens)
    print("helloooo")
    print(tfidf)
    sim_values = cosine_similarity(tfidf[-1], tfidf)
    index = sim_values.argsort()[0][-2]
    flattened_sim = sim_values.flatten()
    flattened_sim.sort()
    required_tfidf = flattened_sim[-2]

    if required_tfidf == 0:
        bot_response += 'Nope'
    else:
        bot_response += sentence_tokens[index]
    return bot_response

flag = 1
while flag == 1:
    user_query = input()
    res = respond(user_query)
    print(res)
