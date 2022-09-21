import math
from nltk.util import ngrams


# each line is a list element
def read_from_file(filepath):
    with open(filepath, "r") as f:
        raw_data = f.readlines()
    return raw_data


# NGRAMS **********************************************************
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


# UNKNOWNS *******************************************************
def get_unknown_ngrams(ngram_dict, n, limit):
    key = '<UNK>' if n == 1 else ('<UNK>', '<UNK>')
    unknowns = {key:value for key, value in ngram_dict.items() if value < limit}
    ngram_dict[key] = sum(unknowns.values())
    return ngram_dict # , unknowns


# SMOOTHING FUNCTIONS *****************************************************
def smoothed_unigram_probabilities(unigrams_dict):
    N = sum(unigrams_dict.values())
    k = 1
    V = len(unigrams_dict)
    smoothed = {}
    for unigram in unigrams_dict:
        smoothed[unigram] = (unigrams_dict[unigram] + k) / (N + (k * V))
    return smoothed


def smoothed_bigram_probabilities(unigrams_dict, bigrams_dict):
    k = 1
    V = len(unigrams_dict)
    smoothed = {}
    for bigram in bigrams_dict:
        smoothed[bigram] = (bigrams_dict[bigram] + k) / (unigrams_dict[bigram[0]] + (k * V))
    return smoothed


# PERPLEXITY ******************************************************
def get_ngram_perplexity(record, trained_probabilities, n):
    summation = 0
    words = record.lower().split()
    M = len(words)
    ngram_list = list(ngrams(words, n)) if n > 1 else words
    unk_key = '<UNK>' if n == 1 else ('<UNK>', '<UNK>')
    for ngram in ngram_list:
        if ngram in trained_probabilities:
            summation += math.log2(trained_probabilities[ngram])
        else:
            summation += math.log2(trained_probabilities[unk_key])
    l = summation / M
    return math.pow(2, (-1) * l)


if __name__ == "__main__":
    # TRAINING
    tr_train_data = read_from_file('A1_DATASET\\train\\truthful.txt')
    de_train_data = read_from_file('A1_DATASET\\train\\deceptive.txt')

    # truthful preprocessing
    tr_unigrams_dict = get_unigrams(tr_train_data)
    tr_bigrams_dict = get_ngrams(tr_train_data, 2)

    # deceptive preprocessing
    de_unigrams_dict = get_unigrams(de_train_data)
    de_bigrams_dict = get_ngrams(de_train_data, 2)

    # UNKNOWN WORD HANDLING
    updated_tr_unigrams = get_unknown_ngrams(tr_unigrams_dict, 1, 7)
    updated_tr_bigrams = get_unknown_ngrams(tr_bigrams_dict, 2, 7)
    updated_de_unigrams = get_unknown_ngrams(de_unigrams_dict, 1, 3)
    updated_de_bigrams = get_unknown_ngrams(de_bigrams_dict, 2, 3)    
    
    # SMOOTHING
    smoothed_tu = smoothed_unigram_probabilities(updated_tr_unigrams)
    smoothed_tb = smoothed_bigram_probabilities(updated_tr_unigrams, updated_tr_bigrams)
    smoothed_du = smoothed_unigram_probabilities(updated_de_unigrams)
    smoothed_db = smoothed_bigram_probabilities(updated_de_unigrams, updated_de_bigrams)

    # VALIDATION ***************************************************************************
    val_tr_data = read_from_file('A1_DATASET\\validation\\truthful.txt')
    val_de_data = read_from_file('A1_DATASET\\validation\\deceptive.txt')

    val_tr_uni_perps = {}
    val_tr_bi_perps = {}
    val_tr_uni_predictions = []
    val_tr_bi_predictions = []
    for i, line in enumerate(val_tr_data):
        val_tr_uni_perps[i] = {'TRUTH': get_ngram_perplexity(line, smoothed_tu, 1),
                               'DECEP': get_ngram_perplexity(line, smoothed_du, 1)}
        val_tr_bi_perps[i] = {'TRUTH': get_ngram_perplexity(line, smoothed_tb, 2),
                               'DECEP': get_ngram_perplexity(line, smoothed_db, 2)}
        if val_tr_uni_perps[i]['TRUTH'] < val_tr_uni_perps[i]['DECEP']:
            val_tr_uni_predictions.append('truthful')
        else:
            val_tr_uni_predictions.append('deceptive')
        if val_tr_bi_perps[i]['TRUTH'] < val_tr_bi_perps[i]['DECEP']:
            val_tr_bi_predictions.append('truthful')
        else:
            val_tr_bi_predictions.append('deceptive')
  
    print(f'\nVALIDATION TRUTHFUL PREDICTIONS ({len(val_tr_uni_predictions)})')
    tu_tr_count = val_tr_uni_predictions.count('truthful')
    tu_de_count = val_tr_uni_predictions.count('deceptive')
    print(f'Unigrams:\ttruthful = {tu_tr_count}\tdeceptive = {tu_de_count}')
    tb_tr_count = val_tr_bi_predictions.count('truthful')
    tb_de_count = val_tr_bi_predictions.count('deceptive')
    print(f'Bigrams:\ttruthful = {tb_tr_count}\tdeceptive = {tb_de_count}')

    val_de_uni_perps = {}
    val_de_bi_perps = {}
    val_de_uni_predictions = []
    val_de_bi_predictions = []
    for i, line in enumerate(val_de_data):
        val_de_uni_perps[i] = {'TRUTH': get_ngram_perplexity(line, smoothed_tu, 1),
                               'DECEP': get_ngram_perplexity(line, smoothed_du, 1)}
        val_de_bi_perps[i] = {'TRUTH': get_ngram_perplexity(line, smoothed_tb, 2),
                               'DECEP': get_ngram_perplexity(line, smoothed_db, 2)}
        if val_de_uni_perps[i]['TRUTH'] < val_de_uni_perps[i]['DECEP']:
            val_de_uni_predictions.append('truthful')
        else:
            val_de_uni_predictions.append('deceptive')
        if val_de_bi_perps[i]['TRUTH'] < val_de_bi_perps[i]['DECEP']:
            val_de_bi_predictions.append('truthful')
        else:
            val_de_bi_predictions.append('deceptive')

    print(f'\nVALIDATION DECEPTIVE PREDICTIONS ({len(val_de_uni_predictions)})')
    du_tr_count = val_de_uni_predictions.count('truthful')
    du_de_count = val_de_uni_predictions.count('deceptive')
    print(f'Unigrams:\ttruthful = {du_tr_count}\tdeceptive = {du_de_count}')
    db_tr_count = val_de_bi_predictions.count('truthful')
    db_de_count = val_de_bi_predictions.count('deceptive')
    print(f'Bigrams:\ttruthful = {db_tr_count}\tdeceptive = {db_de_count}')