import math
from nltk import word_tokenize
from nltk.util import ngrams


# each line is a list element
def read_from_file(filepath):
    with open(filepath, "r") as f:
        raw_data = f.readlines()
    return raw_data


# returns list data split into train & validate lists
def data_split(raw_data, train_portion):
    i_split = math.floor(len(raw_data) * train_portion)
    return raw_data[:i_split], raw_data[i_split:]


# processes raw data and return unigrams dictionary
# dict -> key = unigram, value = occurrence count
def get_unigrams(raw_data):
    unigrams_dict = {}
    for text in raw_data:
        tokens = word_tokenize(text)
        for token in set(tokens):
            if token in unigrams_dict:
                unigrams_dict[token] += tokens.count(token)
            else:
                unigrams_dict[token] = tokens.count(token)
    return unigrams_dict


# processes raw data and returns n-grams dictionary
# dict -> key = n-gram tuple, value = occurrence count
def get_ngrams(raw_data, n):
    ngrams_dict = {}
    for text in raw_data:
        ngrams_list = list(ngrams(word_tokenize(text), n))
        for n_gram in ngrams_list:
            if n_gram in ngrams_dict:
                ngrams_dict[n_gram] += ngrams_list.count(n_gram)
            else:
                ngrams_dict[n_gram] = ngrams_list.count(n_gram)
    return ngrams_dict


# Probability(unigram) = count(unigram) / len(unigrams)
def calc_unigram_probs(unigrams_dict):
    probabilities = {}
    vocab_len = sum(unigrams_dict.values())
    for unigram in unigrams_dict:
        probabilities[unigram] = unigrams_dict[unigram] / vocab_len
    return probabilities


# Probability(bigram) = count(bigram) / count(bigram[0])
def calc_bigram_probs(unigrams_dict, bigrams_dict):
    probabilities = {}
    for bigram in bigrams_dict:
        probabilities[bigram] = bigrams_dict[bigram] / unigrams_dict[bigram[0]]
    return probabilities


# TEMP SMOOTHING FUNCTION
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

# gets keys not in vocabulary and total count of occurrences
def get_unknown(vocab_dict, observed_dict):
    unknowns = observed_dict.keys() - vocab_dict.keys()
    unk_dict = {unk:observed_dict[unk] for unk in unknowns}
    unk_prob = sum(unk_dict.values()) / sum(observed_dict.values())
    return unk_dict, unk_prob


# TEMP PERPLEXITY FUNCTION
def get_perplexity(raw_data, unigram_probs, unknown_prob):
    perplexities = []
    for data in raw_data:
        summation = 0
        words = data.split()
        for word in words:
            if word in unigram_probs:
                summation += math.log2(unigram_probs[word])
            else:
                summation += math.log2(unknown_prob)
        l_value = summation / len(words)
        perplexities.append(math.pow(2, (-1) * l_value))
    return perplexities


if __name__ == "__main__":
    # TRAINING
    truth_data = read_from_file('A1_DATASET\\train\\truthful.txt')
    deceptive_data = read_from_file('A1_DATASET\\train\\deceptive.txt')

    # truthful modeling
    t_unigrams_dict = get_unigrams(truth_data)
    t_bigrams_dict = get_ngrams(truth_data, 2)
    tu_probs = calc_unigram_probs(t_unigrams_dict)
    tb_probs = calc_bigram_probs(t_unigrams_dict, t_bigrams_dict)

    # deceptive modeling
    d_unigrams_dict = get_unigrams(deceptive_data)
    d_bigrams_dict = get_ngrams(deceptive_data, 2)
    du_probs = calc_unigram_probs(d_unigrams_dict)
    db_probs = calc_bigram_probs(d_unigrams_dict, d_bigrams_dict)

    # SMOOTHING
    smoothed_tu = smoothed_unigram_probabilities(t_unigrams_dict)
    smoothed_tb = smoothed_bigram_probabilities(t_unigrams_dict, t_bigrams_dict)
    smoothed_du = smoothed_unigram_probabilities(d_unigrams_dict)
    smoothed_db = smoothed_bigram_probabilities(d_unigrams_dict, d_bigrams_dict)

    # VALIDATION
    val_tr_data = read_from_file('A1_DATASET\\validation\\truthful.txt')
    val_tr_unigrams = get_unigrams(val_tr_data)
    val_tr_bigrams = get_ngrams(val_tr_data, 2)
    val_tu_probs = calc_unigram_probs(val_tr_unigrams)
    val_tb_probs = calc_bigram_probs(val_tr_unigrams, val_tr_bigrams)

    val_de_data = read_from_file('A1_DATASET\\validation\\deceptive.txt')
    val_de_unigrams = get_unigrams(val_de_data)
    val_de_bigrams = get_ngrams(val_de_data, 2)
    val_du_probs = calc_unigram_probs(val_de_unigrams)
    val_db_probs = calc_bigram_probs(val_de_unigrams, val_de_bigrams)

    # UNKNOWN HANDLING
    unk_tr_uni_dict, unk_tr_uni_prob = get_unknown(t_unigrams_dict, val_tr_unigrams)
    unk_tr_bi_dict, unk_tr_bi_prob = get_unknown(t_bigrams_dict, val_tr_bigrams)

    unk_de_uni_dict, unk_de_uni_prob = get_unknown(d_unigrams_dict, val_de_unigrams)
    unk_de_bi_dict, unk_de_bi_prob = get_unknown(d_bigrams_dict, val_de_bigrams)

    # PERPLEXITY
    val_tr_perplexity = get_perplexity(val_tr_data, val_tu_probs, unk_tr_uni_prob)
    val_de_perplexity = get_perplexity(val_de_data, val_du_probs, unk_de_uni_prob)

    print('UNSMOOTHED PERPLEXITY')
    print(f'\nAverage Validation Truth UNK Probability:\nUnigram={unk_tr_uni_prob}\nBigram={unk_tr_bi_prob}')
    print(f'\nAverage Validation Truth Perplexity: {sum(val_tr_perplexity) / len(val_tr_perplexity)}')

    print(f'\nAverage Validation Deceptive UNK Probability:\nUnigram={unk_de_uni_prob}\nBigram={unk_de_bi_prob}')
    print(f'\nAverage Validation Deceptive Perplexity: {sum(val_de_perplexity) / len(val_de_perplexity)}')

    # PREDICTIONS
