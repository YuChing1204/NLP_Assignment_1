from nltk import bigrams
from nltk.tokenize import word_tokenize


def read_file(path):
    text = ""
    with open(path) as f:
        for line in f.readlines():
            text += line
    return text


def tokens_process(data):
    tokens = word_tokenize(data)

    new_tokens = []
    for token in tokens:
        if token != ',' and token != '.' and token != '!':
            new_tokens.append(token)

    return new_tokens

def build_uni_vocabulary(tokens_list):
    vocabulary_set = {}
    for tokens in tokens_list:
        for token in tokens:
            vocabulary_set[token] = 0

    return vocabulary_set

def build_bi_vocabulary(tokens_list):
    vocabulary_set = {}
    for tokens in tokens_list:
        bigram = bigrams(tokens)
        for bi in bigram:
            vocabulary_set[bi] = 0

    return vocabulary_set
def count_unigram(tokens, vocabulary_set):
    vocabulary_set = vocabulary_set.copy()
    for token in tokens:
        vocabulary_set[token] += 1

    return vocabulary_set


def prob_unigram(tokens, vocabulary_set):
    unigram_dic = count_unigram(tokens, vocabulary_set)
    prob_unigram_dic = {}
    for unigram in unigram_dic:
        prob_unigram_dic[unigram] = unigram_dic[unigram] / sum(unigram_dic.values())

    return prob_unigram_dic


def count_bigram(tokens, vocabulary_set):
    vocabulary_set = vocabulary_set.copy()
    bigram = bigrams(tokens)
    for bi in bigram:
        vocabulary_set[bi] += 1

    return vocabulary_set


def prob_bigram(tokens, vocabulary_uni_set, vocabulary_bi_set):
    unigram_dic = count_unigram(tokens, vocabulary_uni_set)
    bigram_dic = count_bigram(tokens, vocabulary_bi_set)
    prob_bigram_dic = {}
    for bigram in bigram_dic:
        if bigram_dic[bigram] != 0:
            prob_bigram_dic[bigram] = bigram_dic[bigram] / unigram_dic[bigram[0]]

    return prob_bigram_dic



