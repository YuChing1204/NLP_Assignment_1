import math
from nltk.util import ngrams


# each line is a list element
def read_from_file(filepath):
    with open(filepath, "r") as f:
        raw_data = f.readlines()
    return raw_data


# processes raw data and return unigrams dictionary
# dict -> key = unigram, value = occurrence count
def get_unigrams(raw_data):
    unigrams_dict = {}
    for text in raw_data:
        tokens = text.split()
        tokens.insert(0, '<s>')
        tokens.append('</s>')
        for token in set(tokens):
            temp = token.lower()
            if temp in unigrams_dict:
                unigrams_dict[temp] += tokens.count(temp)
            else:
                unigrams_dict[temp] = tokens.count(temp)
    return unigrams_dict


# processes raw data and returns n-grams dictionary
# dict -> key = n-gram tuple, value = occurrence count
def get_ngrams(raw_data, n):
    ngrams_dict = {}
    for text in raw_data:
        ngrams_list = list(ngrams(text.split(), n, pad_left=True,
                                                   left_pad_symbol='<s>',
                                                   pad_right=True,
                                                   right_pad_symbol='</s>'))
        for n_gram in ngrams_list:
            temp = (n_gram[0].lower(), n_gram[1].lower())
            if temp in ngrams_dict:
                ngrams_dict[temp] += ngrams_list.count(temp)
            else:
                ngrams_dict[temp] = ngrams_list.count(temp)
    return ngrams_dict


# Probability(unigram) = count(unigram) / len(unigrams)
# def calc_unigram_probs(unigrams_dict):
#     probabilities = {}
#     vocab_len = sum(unigrams_dict.values())
#     for unigram in unigrams_dict:
#         probabilities[unigram] = unigrams_dict[unigram] / vocab_len
#     return probabilities


# # Probability(bigram) = count(bigram) / count(bigram[0])
# def calc_bigram_probs(unigrams_dict, bigrams_dict):
#     probabilities = {}
#     for bigram in bigrams_dict:
#         probabilities[bigram] = bigrams_dict[bigram] / unigrams_dict[bigram[0]]
#     return probabilities


# TEMP SMOOTHING FUNCTIONS **********************************************************
def smoothed_unigram_probabilities(unigrams_dict):
    V = len(unigrams_dict)
    k = 1
    smoothed = {}
    vocab_len = sum(unigrams_dict.values())
    for unigram in unigrams_dict:
        smoothed[unigram] = (unigrams_dict[unigram] + k) / (vocab_len + (k * V))
    return smoothed

def smoothed_bigram_probabilities(unigrams_dict, bigrams_dict):
    V = len(unigrams_dict)
    k = 1
    smoothed = {}
    for bigram in bigrams_dict:
        smoothed[bigram] = (bigrams_dict[bigram] + k) / (unigrams_dict[bigram[0]] + (k * V))
    return smoothed
# TEMP SMOOTHING FUNCTIONS **********************************************************

# gets keys not in vocabulary and total count of occurrences
def get_unknown_words(vocab_dict, observed_dict):
    unknowns = observed_dict.keys() - vocab_dict.keys()
    unk_dict = {unk:observed_dict[unk] for unk in unknowns}
    unk_prob = sum(unk_dict.values()) / sum(observed_dict.values())
    return unk_dict, unk_prob

def get_unknown_ngrams(vocab_dict, observed_dict, unk_word_count):
    unknowns = observed_dict.keys() - vocab_dict.keys()
    unk_ngram_dict = {}
    for unknown in unknowns:
        if unknown[0] in vocab_dict:
            unk_ngram_dict[unknown] = observed_dict[unknown] / vocab_dict[unknown[0]]
        else:
            unk_ngram_dict[unknown] = observed_dict[unknown] / unk_word_count
    unk_ngram_prob = sum(unk_ngram_dict.values()) / len(unk_ngram_dict)
    return unk_ngram_dict, unk_ngram_prob


# TEMP PERPLEXITY FUNCTIONS ******************************************
# def get_perplexity(raw_data, unigram_probs, unknown_prob):
#     perplexities = []
#     for data in raw_data:
#         summation = 0
#         words = data.split()
#         for word in words:
#             if word in unigram_probs:
#                 summation += math.log2(unigram_probs[word])
#             else:
#                 summation += math.log2(unknown_prob)
#         l_value = summation / len(words)
#         perplexities.append(math.pow(2, (-1) * l_value))
#     return perplexities

def get_ngram_perplexity(raw_data, ngram_probs, n, unknown_prob):
    perplexities = []
    for data in raw_data:
        summation = 0
        if n > 1:
            ngram_list = list(ngrams(data.split(), n))
        else:
            ngram_list = data.split()
        for ngram in ngram_list:
            if ngram in ngram_probs:
                summation += math.log2(ngram_probs[ngram])
            else:
                summation += math.log2(unknown_prob)
        l_value = summation / len(ngram_list)
        perplexities.append(math.pow(2, (-1) * l_value))
    return perplexities
# TEMP PERPLEXITY FUNCTIONS ******************************************


if __name__ == "__main__":
    # TRAINING
    truth_data = read_from_file('A1_DATASET\\train\\truthful.txt')
    deceptive_data = read_from_file('A1_DATASET\\train\\deceptive.txt')

    # truthful modeling
    t_unigrams_dict = get_unigrams(truth_data)
    t_bigrams_dict = get_ngrams(truth_data, 2)
    # tu_probs = calc_unigram_probs(t_unigrams_dict)
    # tb_probs = calc_bigram_probs(t_unigrams_dict, t_bigrams_dict)

    # deceptive modeling
    d_unigrams_dict = get_unigrams(deceptive_data)
    d_bigrams_dict = get_ngrams(deceptive_data, 2)
    # du_probs = calc_unigram_probs(d_unigrams_dict)
    # db_probs = calc_bigram_probs(d_unigrams_dict, d_bigrams_dict)

    # SMOOTHING
    smoothed_tu = smoothed_unigram_probabilities(t_unigrams_dict)
    smoothed_tb = smoothed_bigram_probabilities(t_unigrams_dict, t_bigrams_dict)
    smoothed_du = smoothed_unigram_probabilities(d_unigrams_dict)
    smoothed_db = smoothed_bigram_probabilities(d_unigrams_dict, d_bigrams_dict)

    # VALIDATION ***************************************************************************
    val_tr_data = read_from_file('A1_DATASET\\validation\\truthful.txt')
    val_tr_unigrams = get_unigrams(val_tr_data)
    val_tr_bigrams = get_ngrams(val_tr_data, 2)
    # val_tu_probs = calc_unigram_probs(val_tr_unigrams)
    # val_tb_probs =  calc_bigram_probs(val_tr_unigrams, val_tr_bigrams)
    smooth_val_tu_probs = smoothed_unigram_probabilities(val_tr_unigrams)
    smooth_val_tb_probs = smoothed_bigram_probabilities(val_tr_unigrams, val_tr_bigrams)

    val_de_data = read_from_file('A1_DATASET\\validation\\deceptive.txt')
    val_de_unigrams = get_unigrams(val_de_data)
    val_de_bigrams = get_ngrams(val_de_data, 2)
    # val_du_probs = calc_unigram_probs(val_de_unigrams)
    # val_db_probs = calc_bigram_probs(val_de_unigrams, val_de_bigrams)
    smooth_val_du_probs = smoothed_unigram_probabilities(val_de_unigrams)
    smooth_val_db_probs = smoothed_bigram_probabilities(val_de_unigrams, val_de_bigrams)

    # UNKNOWN HANDLING ******************************************************************************
    unk_tr_uni_dict, unk_tr_uni_prob = get_unknown_words(t_unigrams_dict, val_tr_unigrams)
    # unk_tr_bi_dict, unk_tr_bi_prob = get_unknown(t_bigrams_dict, val_tr_bigrams)
    unk_tr_words = sum(unk_tr_uni_dict.values())
    unk_tr_bi_dict, unk_tr_bi_prob = get_unknown_ngrams(t_bigrams_dict, val_tr_bigrams, unk_tr_words)

    unk_de_uni_dict, unk_de_uni_prob = get_unknown_words(d_unigrams_dict, val_de_unigrams)
    # unk_de_bi_dict, unk_de_bi_prob = get_unknown(d_bigrams_dict, val_de_bigrams)
    unk_de_words = sum(unk_de_uni_dict.values())
    unk_de_bi_dict, unk_de_bi_prob = get_unknown_ngrams(d_bigrams_dict, val_de_bigrams, unk_de_words)

    print('\nUNKNOWN PROBABILITIES')
    print('\nAverage Validation Truth UNK Probability')
    print(f'Unigram = {unk_tr_uni_prob}\nBigram = {unk_tr_bi_prob}')
    print('\nAverage Validation Deceptive UNK Probability:')
    print(f'Unigram = {unk_de_uni_prob}\nBigram={unk_de_bi_prob}')

    # PERPLEXITY - UNSMOOTHED VALIDATION DATA ***********************************************
    # val_tr_perplexity = get_ngram_perplexity(val_tr_data, val_tu_probs, 1, unk_tr_uni_prob)
    # val_de_perplexity = get_ngram_perplexity(val_de_data, val_du_probs, 1, unk_de_uni_prob)
    # vtr_bi_perp = get_ngram_perplexity(val_tr_data, val_tb_probs, 2, unk_tr_bi_prob)
    # vde_bi_perp = get_ngram_perplexity(val_de_data, val_db_probs, 2, unk_de_bi_prob)

    # print('\nUNSMOOTHED PERPLEXITY')
    # print(f'\nAverage Validation Truth Perplexity: {sum(val_tr_perplexity) / len(val_tr_perplexity)}')
    # print(f'Average Validation Deceptive Perplexity: {sum(val_de_perplexity) / len(val_de_perplexity)}')
    # print(f'Avg Truth Bigram Perplexity: {sum(vtr_bi_perp) / len(vtr_bi_perp)}')
    # print(f'Avg Deceprive Bigram Perplexity: {sum(vde_bi_perp) / len(vde_bi_perp)}')

    # PERPLEXITY - SMOOTHED VALIDATION DATA ************************************************
    sm_val_tu_perplexity = get_ngram_perplexity(val_tr_data, smooth_val_tu_probs, 1, unk_tr_uni_prob)
    sm_val_du_perplexity = get_ngram_perplexity(val_de_data, smooth_val_du_probs, 1, unk_de_uni_prob)
    sm_val_tb_perplexity = get_ngram_perplexity(val_tr_data, smooth_val_tb_probs, 2, unk_tr_bi_prob)
    sm_val_db_perplexity = get_ngram_perplexity(val_de_data, smooth_val_db_probs, 2, unk_de_bi_prob)

    print('\nSMOOTHED PERPLEXITY')
    print(f'\nAvg Validation Truth Unigram Perplexity: {sum(sm_val_tu_perplexity) / len(sm_val_tu_perplexity)}')
    print(f'Avg Validation Deceptive Unigram Perplexity: {sum(sm_val_du_perplexity) / len(sm_val_du_perplexity)}')
    print(f'\nAvg Validation Truth Bigram Perplexity: {sum(sm_val_tb_perplexity) / len(sm_val_tb_perplexity)}')
    print(f'Avg Validation Deceptive Bigram Perplexity: {sum(sm_val_db_perplexity) / len(sm_val_db_perplexity)}')

    # PREDICTIONS
