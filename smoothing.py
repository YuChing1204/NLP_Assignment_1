import ngrams_model


def prob_laplace_smooth_unigram(count_unigram):
    len_unigram_dic = len(count_unigram)
    prob_laplace_dic = {}

    for unigram in count_unigram:
        count_unigram[unigram] += 1

    for unigram in count_unigram:
        prob_laplace_dic[unigram] = ((count_unigram[unigram] + 1) / (sum(count_unigram.values()) + len_unigram_dic * 1))

    return prob_laplace_dic

def prob_laplace_smooth_bigram(count_unigram, count_bigram):
    len_unigram_dic = len(count_unigram)
    len_bigram_dic = len(count_bigram)
    prob_laplace_dic = {}

    for bigram in count_bigram:
        # if bigram[0] in count_unigram:
        prob_laplace_dic[bigram] = ((count_bigram[bigram] + 1) / (count_unigram[bigram[0]] + len_unigram_dic * 1))
        # else:
        #     print("***")
        #     prob_laplace_dic[bigram] = ((count_bigram[bigram] + 1) / (count_unigram["<unk>"] + len_unigram_dic * 1))

    return prob_laplace_dic
def prob_add_k_smooth_unigram(count_unigram, k):
    len_unigram_dic = len(count_unigram)
    prob_laplace_dic = {}

    for unigram in count_unigram:
        count_unigram[unigram] += 1

    for unigram in count_unigram:
        prob_laplace_dic[unigram] = ((count_unigram[unigram] + k) / (sum(count_unigram.values()) + len_unigram_dic * k))

    return prob_laplace_dic


def prob_add_k_smooth_bigram(count_unigram, count_bigram, k):
    len_unigram_dic = len(count_unigram)
    len_bigram_dic = len(count_bigram)
    prob_laplace_dic = {}

    for bigram in count_bigram:
        # if bigram[0] in count_unigram:
        #     prob_laplace_dic[bigram] = ((count_bigram[bigram] + k) / (count_unigram[bigram[0]] + len_unigram_dic * k))
        # else:
        prob_laplace_dic[bigram] = ((count_bigram[bigram] + k) / (count_unigram["<unk>"] + len_unigram_dic * k))

    return prob_laplace_dic
