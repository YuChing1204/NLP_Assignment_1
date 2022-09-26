import itertools

import nltk
from nltk import bigrams
from nltk.tokenize import word_tokenize


def read_file(path):
    text = []
    with open(path) as f:
        for line in f.readlines():
            tokenizer = nltk.RegexpTokenizer(r"\w+")
            new_words = tokenizer.tokenize(line)

            text.append("<s>")
            for word in new_words:
                text.append(word)
            text.append("</s>")

    return text


def tokens_process(line):
    text = []
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    new_words = tokenizer.tokenize(line)

    text.append("<s>")

    for word in new_words:
        text.append(word)

    text.append("</s>")
    return text

def build_uni_vocabulary(tokens_list):
    vocabulary_set = {}
    for tokens in tokens_list:
        for token in tokens:
            vocabulary_set[token] = 0

    return vocabulary_set

def build_bi_vocabulary(uni_vocabulary):
    vocabulary_set = {}
    combs = itertools.product(uni_vocabulary, repeat = 2)
    for bigram in combs:
        vocabulary_set[bigram] = 0

    return vocabulary_set
def count_unigram(tokens):
    vocabulary_set = {}
    for token in tokens:
        if token in vocabulary_set:
            vocabulary_set[token] += 1
        else:
            vocabulary_set[token] = 1
    return vocabulary_set


def prob_unigram(count_unigram):
    prob_unigram_dic = {}
    for unigram in count_unigram:
        prob_unigram_dic[unigram] = count_unigram[unigram] / sum(count_unigram.values())
    return prob_unigram_dic


def count_bigram(tokens, vocabulary_set):
    vocabulary_set = vocabulary_set.copy()
    bi_grams = bigrams(tokens)
    for bigram in bi_grams:
        vocabulary_set[bigram] += 1

    return vocabulary_set


def prob_bigram(count_bigram, count_unigram):
    prob_bigram_dic = {}
    for bigram in count_bigram:
        prob_bigram_dic[bigram] = count_bigram[bigram] / count_unigram[bigram[0]]

    return prob_bigram_dic



