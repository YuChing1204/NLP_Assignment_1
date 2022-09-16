import math
from nltk import word_tokenize
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


# gets keys not in vocabulary and total count of occurrences
def get_unknown(vocab_dict, observed_dict):
    unknown = observed_dict.keys() - vocab_dict.keys()
    unk_count = sum([observed_dict[key] for key in unknown])
    return unknown, unk_count


# returns list data split into train & validate lists
def data_split(raw_data, train_portion):
    i_split = math.floor(len(raw_data) * train_portion)
    return raw_data[:i_split], raw_data[i_split:]


if __name__ == "__main__":
    # PROCESSING
    truth_data = read_from_file('A1_DATASET\\train\\truthful.txt')
    train_t_data, val_t_data = data_split(truth_data, 0.8)

    deceptive_data = read_from_file('A1_DATASET\\train\\deceptive.txt')
    train_d_data, val_d_data = data_split(deceptive_data, 0.8)

    # truthful modeling
    t_unigrams_dict = get_unigrams(train_t_data)
    t_bigrams_dict = get_ngrams(train_t_data, 2)
    tu_probs = calc_unigram_probs(t_unigrams_dict)
    tb_probs = calc_bigram_probs(t_unigrams_dict, t_bigrams_dict)

    # deceptive modeling
    d_unigrams_dict = get_unigrams(train_d_data)
    d_bigrams_dict = get_ngrams(train_d_data, 2)
    du_probs = calc_unigram_probs(d_unigrams_dict)
    db_probs = calc_bigram_probs(d_unigrams_dict, d_bigrams_dict)

     
    # SMOOTHING
    # PERPLEXITY
    # PREDICTIONS
