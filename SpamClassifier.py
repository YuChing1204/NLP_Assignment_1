import math
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import string

# each line is a list element
def read_from_file(filepath):
    raw_data = []
    with open(filepath, "r") as f:
        for line in f:
            raw_data.append(line.strip())
    return raw_data

# each line is a list element
def labels_from_file(filepath):
    raw_data = []
    with open(filepath, "r") as f:
        for line in f:
            raw_data.append(int(line.strip()))
    return raw_data


def preprocess_text(line):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(line.lower())
    stops = set(stopwords.words('english'))
    tokens = [word for word in words if word not in stops]
    tokens.insert(0, '<s>')
    tokens.append('</s>')
    return tokens


# NGRAMS **********************************************************
# processes raw data and return unigrams dictionary
# dict -> key = unigram, value = occurrence count
def get_unigrams(raw_data):
    unigrams_dict = {}
    for text in raw_data:
        tokens = preprocess_text(text)
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
        ngrams_list = list(ngrams(preprocess_text(text), n))
        for n_gram in ngrams_list:
            if n_gram in ngrams_dict:
                ngrams_dict[n_gram] += ngrams_list.count(n_gram)
            else:
                ngrams_dict[n_gram] = ngrams_list.count(n_gram)
    return ngrams_dict


# UNKNOWNS *******************************************************
def get_unknown_ngrams(ngram_dict, n, limit):
    key = '<UNK>' if n == 1 else ('<UNK>', '<UNK>')
    unknowns = {key:value for key, value in ngram_dict.items() if value <= limit}
    ngram_dict[key] = sum(unknowns.values())
    return ngram_dict # , unknowns


# SMOOTHING FUNCTIONS *****************************************************
def smoothed_unigram_probabilities(unigrams_dict, k):
    N = sum(unigrams_dict.values())
    V = len(unigrams_dict)
    smoothed = {}
    for unigram in unigrams_dict:
        smoothed[unigram] = (unigrams_dict[unigram] + k) / (N + (k * V))
    return smoothed


def smoothed_bigram_probabilities(unigrams_dict, bigrams_dict, k):
    V = len(bigrams_dict)
    smoothed = {}
    for bigram in bigrams_dict:
        smoothed[bigram] = (bigrams_dict[bigram] + k) / (unigrams_dict[bigram[0]] + (k * V))
    return smoothed


# PERPLEXITY ******************************************************
def get_ngram_perplexity(record, trained_probabilities, n):
    summation = 0
    words = preprocess_text(record)
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
    avg_perp = 0
    for line in test_data:
        truth_score = get_ngram_perplexity(line, truthful_model, n)
        deceptive_score = get_ngram_perplexity(line, deceptive_model, n)
        if truth_score < deceptive_score:
            predictions.append(0)
            avg_perp += truth_score
        else:
            predictions.append(1)
            avg_perp += deceptive_score
    ngram_type = 'unigram' if n == 1 else 'bigram'
    print(f'\n{ngram_type} avg perplexity: {avg_perp / len(predictions)}')
    return predictions


def check_accuracy(predictions, actuals):
    accurate = 0
    for predicted, actual in zip(predictions, actuals):
        if predicted == actual:
            accurate += 1
    return accurate / len(predictions) * 100.0


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
    updated_tr_unigrams = get_unknown_ngrams(tr_unigrams_dict, 1, 15)
    updated_tr_bigrams = get_unknown_ngrams(tr_bigrams_dict, 2, 0)
    updated_de_unigrams = get_unknown_ngrams(de_unigrams_dict, 1, 30)
    updated_de_bigrams = get_unknown_ngrams(de_bigrams_dict, 2, 0)
    # print('Truthful unigram UNK count:  ' + str(updated_tr_unigrams['<UNK>']))
    # print('Truthful bigram UNK count:   ' + str(updated_tr_bigrams[('<UNK>', '<UNK>')]))
    # print('Deceptive unigram UNK count: ' + str(updated_de_unigrams['<UNK>']))
    # print('Deceptive bigram UNK count:  ' + str(updated_de_bigrams[('<UNK>', '<UNK>')])) 
    

    # SMOOTHING
    smoothed_tu = smoothed_unigram_probabilities(updated_tr_unigrams, 1)
    smoothed_tb = smoothed_bigram_probabilities(updated_tr_unigrams, updated_tr_bigrams, 1)
    smoothed_du = smoothed_unigram_probabilities(updated_de_unigrams, 1)
    smoothed_db = smoothed_bigram_probabilities(updated_de_unigrams, updated_de_bigrams, 1)
    
    print(f'smooth truthful uni probability sum = {sum(smoothed_tu.values())}')
    print(f'smooth truthful bi probability sum = {sum(smoothed_tb.values())}')
    print(f'smooth deceptive uni probability sum = {sum(smoothed_du.values())}')
    print(f'smooth deceptive bi probability sum = {sum(smoothed_db.values())}')
    print(f'max truthful bigram prob = {max(smoothed_tb)} {max(smoothed_tb.values())}')
    print(f'max deceptive bigram prob = {max(smoothed_db)} {max(smoothed_db.values())}')
    # print('Truthful unigram UNK prob:  ' + str(smoothed_tu['<UNK>']))
    print('Truthful bigram UNK prob:   ' + str(smoothed_tb[('<UNK>', '<UNK>')]))
    # print('Deceptive unigram UNK prob: ' + str(smoothed_du['<UNK>']))
    print('Deceptive bigram UNK prob:  ' + str(smoothed_db[('<UNK>', '<UNK>')])) 
    
    # VALIDATION ***************************************************************************
    val_tr_data = read_from_file('A1_DATASET\\validation\\truthful.txt')
    val_de_data = read_from_file('A1_DATASET\\validation\\deceptive.txt')
    
    print('\n... VALIDATION DATA BEGINS ...')
  
    print(f'\n****** VALIDATION TRUTHFUL PREDICTIONS ({len(val_tr_data)}) ******')
    val_tr_uni_preds = make_predictions(val_tr_data, smoothed_tu, smoothed_du, 1)
    val_tr_bi_preds = make_predictions(val_tr_data, smoothed_tb, smoothed_db, 2)

    val_tr_labels = [0] * len(val_tr_uni_preds)
    tu_tr_count = val_tr_uni_preds.count(0)
    tu_de_count = val_tr_uni_preds.count(1)
    print(f'\nUnigrams:\ttruthful = {tu_tr_count}\tdeceptive = {tu_de_count}')
    tb_tr_count = val_tr_bi_preds.count(0)
    tb_de_count = val_tr_bi_preds.count(1)
    print(f'Bigrams:\ttruthful = {tb_tr_count}\tdeceptive = {tb_de_count}')
    print(f'\nVAL TRUTHFUL UNIGRAM ACCURACY = {check_accuracy(val_tr_uni_preds, val_tr_labels)}')
    print(f'VAL TRUTHFUL BIGRAM ACCURACY = {check_accuracy(val_tr_bi_preds, val_tr_labels)}')

    print(f'\n****** VALIDATION DECEPTIVE PREDICTIONS ({len(val_de_data)}) ******')
    val_de_uni_preds = make_predictions(val_de_data, smoothed_tu, smoothed_du, 1)
    val_de_bi_preds = make_predictions(val_de_data, smoothed_tb, smoothed_db, 2)
    
    val_de_labels = [1] * len(val_de_uni_preds)
    du_tr_count = val_de_uni_preds.count(0)
    du_de_count = val_de_uni_preds.count(1)
    print(f'\nUnigrams:\ttruthful = {du_tr_count}\tdeceptive = {du_de_count}')
    db_tr_count = val_de_bi_preds.count(0)
    db_de_count = val_de_bi_preds.count(1)
    print(f'Bigrams:\ttruthful = {db_tr_count}\tdeceptive = {db_de_count}')
    print(f'\nVAL DECEPTIVE UNIGRAM ACCURACY = {check_accuracy(val_de_uni_preds, val_de_labels)}')
    print(f'VAL DECEPTIVE BIGRAM ACCURACY = {check_accuracy(val_de_bi_preds, val_de_labels)}')


# TESTING **********************************************************************
    print(f'\n\n... TEST DATA BEGINS ...')
    test_data = read_from_file('A1_DATASET\\test\\test.txt')
    test_labels = labels_from_file('A1_DATASET\\test\\test_labels.txt')

    print(f'\n****** TEST PREDICTIONS ({len(test_labels)}) ******')
    test_uni_preds = make_predictions(test_data, smoothed_tu, smoothed_du, 1)
    test_bi_preds = make_predictions(test_data, smoothed_tb, smoothed_db, 2)

    tu_tr_count = test_uni_preds.count(0)
    tu_de_count = test_uni_preds.count(1)
    print(f'\nUnigrams:\ttruthful = {tu_tr_count}\tdeceptive = {tu_de_count}')
    tb_tr_count = test_uni_preds.count(0)
    tb_de_count = test_uni_preds.count(1)
    print(f'Bigrams:\ttruthful = {tb_tr_count}\tdeceptive = {tb_de_count}')
    print(f'\nTEST UNIGRAM ACCURACY = {check_accuracy(test_uni_preds, test_labels)}')
    print(f'TEST BIGRAM ACCURACY = {check_accuracy(test_bi_preds, test_labels)}')
    
    print(f'\nunigram predictions\n{test_uni_preds}')
    print(f'\nbigram predictions\n{test_bi_preds}')
