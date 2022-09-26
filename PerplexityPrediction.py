import LanguageModel as lm
import math
from nltk.util import ngrams

# PERPLEXITY ******************************************************
def get_ngram_perplexity(line, trained_probabilities, n):
    summation = 0
    words = lm.preprocess_text(line)
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


# PREDICTIONS ******************************************************
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


# ACCURACY *********************************************************
def prediction_accuracy(predictions, actuals):
    accurate = 0
    for predicted, actual in zip(predictions, actuals):
        if predicted == actual:
            accurate += 1
    return accurate / len(predictions) * 100.0
