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
    vocab_len = len(unigrams_dict)
    for unigram in unigrams_dict:
        probabilities[unigram] = unigrams_dict[unigram] / vocab_len
    return probabilities


# Probability(bigram) = count(bigram) / count(bigram[0])
def calc_bigram_probs(unigrams_dict, bigrams_dict):
    probabilities = {}
    for bigram in bigrams_dict:
        probabilities[bigram] = bigrams_dict[bigram] / unigrams_dict[bigram[0]]
    return probabilities


if __name__ == "__main__":
    # PROCESSING
    truth_data = read_from_file('A1_DATASET\\train\\truthful.txt')
    t_unigrams_dict = get_unigrams(truth_data)
    t_bigrams_dict = get_ngrams(truth_data, 2)
    tu_probs = calc_unigram_probs(t_unigrams_dict)
    tb_probs = calc_bigram_probs(t_unigrams_dict, t_bigrams_dict)

    print('\nMost Probable Truthful Unigram and Bigram')
    print(f'total unigrams: {len(t_unigrams_dict)}')
    max_tu_key = max(t_unigrams_dict, key=t_unigrams_dict.get)
    print(f'unigram max: {max_tu_key}\tcount: {t_unigrams_dict[max_tu_key]}\tprob: {tu_probs[max_tu_key]}\n')

    print(f'total bigrams: {len(t_bigrams_dict)}')
    max_tb_key = max(t_bigrams_dict, key=t_bigrams_dict.get)
    print(f'bigram max: {max_tb_key}\tcount: {t_bigrams_dict[max_tb_key]}\tprob: {tb_probs[max_tb_key]}\n')

    deceptive_data = read_from_file('A1_DATASET\\train\\deceptive.txt')
    d_unigrams_dict = get_unigrams(deceptive_data)
    d_bigrams_dict = get_ngrams(deceptive_data, 2)
    du_probs = calc_unigram_probs(d_unigrams_dict)
    db_probs = calc_bigram_probs(d_unigrams_dict, d_bigrams_dict)

    print('\nMost Probable Deceptive Unigram and Bigram')
    print(f'total unigrams: {len(d_unigrams_dict)}')
    max_du_key = max(d_unigrams_dict, key=d_unigrams_dict.get)
    print(f'unigram max: {max_du_key}\tcount: {d_unigrams_dict[max_du_key]}\tprob: {du_probs[max_du_key]}\n')

    print(f'total bigrams: {len(d_bigrams_dict)}')
    max_db_key = max(d_bigrams_dict, key=d_bigrams_dict.get)
    print(f'bigram max: {max_db_key}\tcount: {d_bigrams_dict[max_db_key]}\tprob: {db_probs[max_db_key]}\n')
    
    # SMOOTHING
    # PERPLEXITY
    # PREDICTIONS
