import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from itertools import combinations_with_replacement


# RAW DATA PREPROCESSING ****************************************************
# removal of punctuation and stopwords
# addition of sentence start and stop delimiters
def preprocess_text(line):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(line.lower())
    stops = set(stopwords.words('english'))
    tokens = [word for word in words if word not in stops]
    tokens.insert(0, '<s>')
    tokens.append('</s>')
    return tokens


# NGRAMS **********************************************************
# unigram dict -> key = unigram string, value = frequency
def get_unigrams(raw_data):
    unigrams_dict = {}
    for line in raw_data:
        ngrams_list = preprocess_text(line)

        for unigram in ngrams_list:
            if unigram in unigrams_dict:
                unigrams_dict[unigram] += ngrams_list.count(unigram)
            else:
                unigrams_dict[unigram] = ngrams_list.count(unigram)
    return unigrams_dict


# ngram dict -> key = ngram tuple, value = frequency
# combinations of each token as 0-count vocabulary
def get_bigrams(raw_data, zero_count=True):
    ngrams_dict = {}
    combs = set()
    for text in raw_data:
        tokens = preprocess_text(text)
        ngrams_list = list(ngrams(tokens, 2))
        
        for bigram in ngrams_list:
            if bigram in ngrams_dict:
                ngrams_dict[bigram] += ngrams_list.count(bigram)
            else:
                ngrams_dict[bigram] = ngrams_list.count(bigram)
        
        if zero_count:
            curr_combs = combinations_with_replacement(tokens, 2)
            for comb in curr_combs:
                comb_tuple = tuple(comb)
                if comb_tuple not in ngrams_dict:
                    ngrams_dict[comb_tuple] = 0
                combs.add(tuple(comb))
    return ngrams_dict


# UNKNOWN HANDLING ********************************************************
# creation of unknown counts based on ngram frequency <= limit
def get_unknown_ngrams(ngram_dict, n, limit):
    key = '<UNK>' if n == 1 else ('<UNK>', '<UNK>')
    unknowns = {key:value for key, value in ngram_dict.items() if value <= limit}
    ngram_dict[key] = sum(unknowns.values())
    return ngram_dict


# UNIGRAM PROBABILITY *****************************************************
def unigram_probabilities(unigrams_dict):
    N = sum(unigrams_dict.values())
    probabilities = {}
    for unigram in unigrams_dict:
        probabilities[unigram] = unigrams_dict[unigram] / N
    return probabilities


def bigram_probabilties(bigrams_dict, unigrams_dict):
    probabilties = {}
    for bigram in bigrams_dict:
        probabilties[bigram] = bigrams_dict[bigram] / unigrams_dict[bigram[0]]
    return probabilties


# ADD-K SMOOTHING *********************************************************
# Use k = 1 for Laplace Smoothing
def smooth_bigrams(unigrams_dict, bigrams_dict, k):
    V = len(unigrams_dict)
    smoothed = {}
    for bigram in bigrams_dict:
        smoothed[bigram] = (bigrams_dict[bigram] + k) / (unigrams_dict[bigram[0]] + (k * V))
    return smoothed
