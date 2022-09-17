import ngrams_model


def prob_laplace_smooth_unigram(data):
    unigram_dic, len_tokens = ngrams_model.count_unigram(data)
    len_unigram_dic = len(unigram_dic)
    prob_laplace_dic = {}

    for unigram in unigram_dic:
        prob_laplace_dic[unigram] = ((unigram_dic[unigram] + 1) / (len_tokens + len_unigram_dic * 1))

    return prob_laplace_dic

def prob_laplace_smooth_bigram(data):
    unigram_dic, len_tokens = ngrams_model.count_unigram(data)
    bigram_dic = ngrams_model.count_bigram(data)
    len_bigram_dic = len(bigram_dic)
    prob_laplace_dic = {}

    for bigram in bigram_dic:
        prob_laplace_dic[bigram] = ((bigram_dic[bigram] + 1) / (unigram_dic[bigram[0]] + len_bigram_dic * 1))

    return prob_laplace_dic
def prob_add_k_smooth_unigram(data, k):
    unigram_dic, len_tokens = ngrams_model.count_unigram(data)
    len_unigram_dic = len(unigram_dic)
    prob_laplace_dic = {}

    for unigram in unigram_dic:
        prob_laplace_dic[unigram] = ((unigram_dic[unigram] + k) / (len_tokens + len_unigram_dic * k))

    return prob_laplace_dic

def prob_add_k_smooth_bigram(data, k):
    unigram_dic, len_tokens = ngrams_model.count_unigram(data)
    bigram_dic = ngrams_model.count_bigram(data)
    len_bigram_dic = len(bigram_dic)
    prob_laplace_dic = {}

    for bigram in bigram_dic:
        prob_laplace_dic[bigram] = ((bigram_dic[bigram] + k) / (unigram_dic[bigram[0]] + len_bigram_dic * k))

    return prob_laplace_dic
