import numpy as np
from nltk import word_tokenize, bigrams

import ngrams_model


def cal_perplexity(prob_dic, sentence):
    tokens = ngrams_model.tokens_process(sentence)
    perplexity = 0
    for n_gram in tokens:
        if n_gram in prob_dic:
            perplexity += np.log2(prob_dic[n_gram])
        else:
            perplexity += np.log2(prob_dic["<unk>"])

    return np.power(2, -(perplexity/len(tokens)))

def cal_bi_perplexity(prob_dic, sentence):
    tokens = ngrams_model.tokens_process(sentence)
    # for i in range(len(tokens)):
    #     token = tokens[i]
    #     if token not in uni_prob_dic:
    #         tokens[i] = "<unk>"

    bigram = bigrams(tokens)
    perplexity = 0
    for n_gram in bigram:
        if n_gram in prob_dic:
            perplexity += np.log2(prob_dic[n_gram])
        else:
            perplexity += np.log2(prob_dic[('<unk>', '<unk>')])


    return np.power(2, -(perplexity/len(tokens)))
