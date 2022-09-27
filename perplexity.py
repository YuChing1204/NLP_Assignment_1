import numpy as np
from nltk import word_tokenize, bigrams

import ngrams_model
import unknown


def cal_perplexity(prob_dic, sentence, del_keys):
    tokens = ngrams_model.tokens_process(sentence)
    tokens = unknown.unknown_tokens_process(tokens, del_keys)
    perplexity = 0
    for n_gram in tokens:
        if n_gram in prob_dic:
            perplexity += np.log2(prob_dic[n_gram])
        else:
            perplexity += np.log2(prob_dic["<unk>"])

    return np.power(2, -(perplexity / len(tokens)))


def cal_bi_perplexity(prob_dic, sentence, del_keys):
    tokens = ngrams_model.tokens_process(sentence)
    tokens = unknown.unknown_tokens_process(tokens, del_keys)
    bigram = bigrams(tokens)
    perplexity = 0
    for n_gram in bigram:
        if n_gram in prob_dic:
            perplexity += np.log2(prob_dic[n_gram])
        else:
            perplexity += np.log2(prob_dic[('<unk>', '<unk>')])

    return np.power(2, -(perplexity / len(tokens)))
