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
        # tokens.insert(0, '<s>')
        # tokens.append('</s>')
        for token in set(tokens):
            temp = token # .lower()
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
        ngrams_list = list(ngrams(text.split(), n)) # , pad_left=True,
                                                   # left_pad_symbol='<s>',
                                                   # pad_right=True,
                                                   # right_pad_symbol='</s>'))
        for n_gram in ngrams_list:
            # temp = (n_gram[0].lower(), n_gram[1].lower())
            if n_gram in ngrams_dict:
                ngrams_dict[n_gram] += ngrams_list.count(n_gram)
            else:
                ngrams_dict[n_gram] = ngrams_list.count(n_gram)
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
    k = 2
    V = len(unigrams_dict)
    smoothed = {}
    for unigram in unigrams_dict:
        smoothed[unigram] = (unigrams_dict[unigram] + k) / (N + (k * V))
    return smoothed


def smoothed_bigram_probabilities(unigrams_dict, bigrams_dict):
    k = 2
    V = len(unigrams_dict)
    smoothed = {}
    for bigram in bigrams_dict:
        smoothed[bigram] = (bigrams_dict[bigram] + k) / (unigrams_dict[bigram[0]] + (k * V))
    return smoothed


# PERPLEXITY ******************************************************
def get_ngram_perplexity(record, trained_probabilities, n):
    summation = 0
    words = record.split() # .lower().split()
    # words.insert(0, '<s>')
    # words.append('</s>')
    # stops = set(stopwords.words('english'))
    # updated_words = [word for word in words if word not in stops]
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


def make_predictions(test_data, truthful_model, deceptive_model, n):
    predictions = []
    for line in test_data:
        truth_score = get_ngram_perplexity(line, truthful_model, n)
        deceptive_score = get_ngram_perplexity(line, deceptive_model, n)
        if truth_score < deceptive_score:
            predictions.append('T')
        else:
            predictions.append('D')
    return predictions


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
    updated_tr_unigrams = get_unknown_ngrams(tr_unigrams_dict, 1, 4)
    updated_tr_bigrams = get_unknown_ngrams(tr_bigrams_dict, 2, 5)
    updated_de_unigrams = get_unknown_ngrams(de_unigrams_dict, 1, 9)
    updated_de_bigrams = get_unknown_ngrams(de_bigrams_dict, 2, 5)    
    
    # SMOOTHING
    smoothed_tu = smoothed_unigram_probabilities(updated_tr_unigrams)
    smoothed_tb = smoothed_bigram_probabilities(updated_tr_unigrams, updated_tr_bigrams)
    smoothed_du = smoothed_unigram_probabilities(updated_de_unigrams)
    smoothed_db = smoothed_bigram_probabilities(updated_de_unigrams, updated_de_bigrams)

    # VALIDATION ***************************************************************************
    val_tr_data = read_from_file('A1_DATASET\\validation\\truthful.txt')
    val_de_data = read_from_file('A1_DATASET\\validation\\deceptive.txt')

    val_tr_uni_preds = make_predictions(val_tr_data, smoothed_tu, smoothed_du, 1)
    val_de_uni_preds = make_predictions(val_de_data, smoothed_tu, smoothed_du, 1)
    val_tr_bi_preds = make_predictions(val_tr_data, smoothed_tb, smoothed_db, 2)
    val_de_bi_preds = make_predictions(val_de_data, smoothed_tb, smoothed_db, 2)
  
    print(f'\nVALIDATION TRUTHFUL PREDICTIONS ({len(val_tr_uni_preds)})')
    tu_tr_count = val_tr_uni_preds.count('T')
    tu_de_count = val_tr_uni_preds.count('D')
    print(f'Unigrams:\ttruthful = {tu_tr_count}\tdeceptive = {tu_de_count}')
    tb_tr_count = val_tr_bi_preds.count('T')
    tb_de_count = val_tr_bi_preds.count('D')
    print(f'Bigrams:\ttruthful = {tb_tr_count}\tdeceptive = {tb_de_count}')

    print(f'\nVALIDATION DECEPTIVE PREDICTIONS ({len(val_de_uni_preds)})')
    du_tr_count = val_de_uni_preds.count('T')
    du_de_count = val_de_uni_preds.count('D')
    print(f'Unigrams:\ttruthful = {du_tr_count}\tdeceptive = {du_de_count}')
    db_tr_count = val_de_bi_preds.count('T')
    db_de_count = val_de_bi_preds.count('D')
    print(f'Bigrams:\ttruthful = {db_tr_count}\tdeceptive = {db_de_count}')
