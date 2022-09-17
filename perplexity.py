import numpy as np


def cal_perplexity(prob_dic):
    perplexity = 0
    len_dic = len(prob_dic)
    for n_gram in prob_dic:
        perplexity += np.log2(prob_dic[n_gram])

    return np.power(2, -(perplexity/len_dic))
