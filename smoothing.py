import ngrams_model
def count_laplace_unigram(data):
    unigram_dic, len_tokens = ngrams_model.count_unigram(data)
    for unigram in unigram_dic:
        unigram_dic[unigram] += 1

    return unigram_dic, len_tokens

def prob_laplace_k_smooth_unigram (data, k):
    laplace_unigram_dic, len_tokens = count_laplace_unigram(data)
    len_unigram_dic = len(laplace_unigram_dic)
    prob_laplace_dic = {}

    for unigram in laplace_unigram_dic:
        prob_laplace_dic[unigram] = ((laplace_unigram_dic[unigram] + k) / (len_tokens + len_unigram_dic * k))

    return prob_laplace_dic

# def count_laplace_bigram(data):
#     bigram_dic = ngrams_model.count_bigram(data)
#     for bigram in bigram_dic:
#         bigram_dic[bigram] += 1
#
#     return unigram_dic, len_tokens
#
# def prob_laplace_k_smooth_bigram (data, k):
#     laplace_bigram_dic, len_tokens = count_laplace(data)
#     len_unigram_dic = len(laplace_unigram_dic)
#     prob_laplace_dic = {}
#
#     for unigram in laplace_unigram_dic:
#         prob_laplace_dic[unigram] = ((laplace_unigram_dic[unigram] + k) / (len_tokens + len_unigram_dic * k))
#
#     return prob_laplace_dic
