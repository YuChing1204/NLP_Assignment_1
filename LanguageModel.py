import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams


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
# ngram dict -> key = n-gram tuple, value = frequency
def get_ngrams(raw_data, n):
    ngrams_dict = {}
    for text in raw_data:
        if n > 1:
            ngrams_list = list(ngrams(preprocess_text(text), n))
        else:
            ngrams_list = preprocess_text(text)

        for n_gram in ngrams_list:
            if n_gram in ngrams_dict:
                ngrams_dict[n_gram] += ngrams_list.count(n_gram)
            else:
                ngrams_dict[n_gram] = ngrams_list.count(n_gram)
    return ngrams_dict


# UNKNOWN HANDLING ********************************************************
# creation of unknown counts based on ngram frequency <= limit
def get_unknown_ngrams(ngram_dict, n, limit):
    key = '<UNK>' if n == 1 else ('<UNK>', '<UNK>')
    unknowns = {key:value for key, value in ngram_dict.items() if value <= limit}
    ngram_dict[key] = sum(unknowns.values())
    return ngram_dict


# ADD-K SMOOTHING *********************************************************
# Use k = 1 for Laplace Smoothing
def smooth_unigrams(unigrams_dict, k):
    N = sum(unigrams_dict.values())
    V = len(unigrams_dict)
    smoothed = {}
    for unigram in unigrams_dict:
        smoothed[unigram] = (unigrams_dict[unigram] + k) / (N + (k * V))
    return smoothed


def smooth_bigrams(unigrams_dict, bigrams_dict, k):
    V = len(unigrams_dict)
    smoothed = {}
    for bigram in bigrams_dict:
        smoothed[bigram] = (bigrams_dict[bigram] + k) / (unigrams_dict[bigram[0]] + (k * V))
    return smoothed
