import LanguageModel as lm
import PerplexityPrediction as pp


# each line is a string list element
def read_from_file(filepath):
    string_list = []
    with open(filepath, "r") as f:
        for line in f:
            string_list.append(line.strip())
    return string_list


# each line is an integer list element
def labels_from_file(filepath):
    int_list = []
    with open(filepath, "r") as f:
        for line in f:
            int_list.append(int(line.strip()))
    return int_list


if __name__ == "__main__":
    # TRAINING **********************************************************
    tr_train_data = read_from_file('A1_DATASET\\train\\truthful.txt')
    de_train_data = read_from_file('A1_DATASET\\train\\deceptive.txt')

    # TRAINING UNIGRAMS: PREPROCESSING & UNKNOWN HANDLING
    # Truthful Model
    tr_unigrams_dict = lm.get_unigrams(tr_train_data)
    updated_tr_unigrams = lm.get_unknown_ngrams(tr_unigrams_dict, 1, 1)
    # Deceptive Model
    de_unigrams_dict = lm.get_unigrams(de_train_data)
    updated_de_unigrams = lm.get_unknown_ngrams(de_unigrams_dict, 1, 2)

    # TRAINING BIGRAMS: PREPROCESSING & UNKNOWN HANDLING
    # Truthful Model
    tr_bigrams_dict = lm.get_bigrams(tr_train_data, True)
    updated_tr_bigrams = lm.get_unknown_ngrams(tr_bigrams_dict, 2, 0)
    # Deceptive Model
    de_bigrams_dict = lm.get_bigrams(de_train_data, True)
    updated_de_bigrams = lm.get_unknown_ngrams(de_bigrams_dict, 2, 0)

    # TRAINING UNIGRAMS: PROBABILITIES
    probs_tu = lm.unigram_probabilities(updated_tr_unigrams)
    probs_du = lm.unigram_probabilities(updated_de_unigrams)
    
    # TRAINING BIGRAMS: SMOOTHED PROBABILITIES
    smoothed_tb = lm.smooth_bigrams(updated_tr_unigrams, updated_tr_bigrams, 1)
    smoothed_db = lm.smooth_bigrams(updated_de_unigrams, updated_de_bigrams, 1)
    
    # VALIDATION ***************************************************************************
    val_tr_data = read_from_file('A1_DATASET\\validation\\truthful.txt')
    val_de_data = read_from_file('A1_DATASET\\validation\\deceptive.txt')
    
    # Truthful Data
    # MAKE PREDICTIONS USING UNIGRAM/BIGRAM MODELS
    val_tr_uni_preds = pp.make_predictions(val_tr_data, probs_tu, probs_du, 1)
    val_tr_bi_preds = pp.make_predictions(val_tr_data, smoothed_tb, smoothed_db, 2)

    val_tr_labels = [0] * len(val_tr_uni_preds)
    tu_tr_count = val_tr_uni_preds.count(0)
    tu_de_count = val_tr_uni_preds.count(1)
    tb_tr_count = val_tr_bi_preds.count(0)
    tb_de_count = val_tr_bi_preds.count(1)

    # Deceptive Data
    # MAKE PREDICTIONS USING UNIGRAM/BIGRAM MODELS
    val_de_uni_preds = pp.make_predictions(val_de_data, probs_tu, probs_du, 1)
    val_de_bi_preds = pp.make_predictions(val_de_data, smoothed_tb, smoothed_db, 2)
    
    val_de_labels = [1] * len(val_de_uni_preds)
    du_tr_count = val_de_uni_preds.count(0)
    du_de_count = val_de_uni_preds.count(1)
    db_tr_count = val_de_bi_preds.count(0)
    db_de_count = val_de_bi_preds.count(1)

# TESTING **********************************************************************
    test_data = read_from_file('A1_DATASET\\test\\test.txt')
    test_labels = labels_from_file('A1_DATASET\\test\\test_labels.txt')

    # MAKE PREDICTIONS USING UNIGRAM/BIGRAM MODELS
    test_uni_preds = pp.make_predictions(test_data, probs_tu, probs_du, 1)
    test_bi_preds = pp.make_predictions(test_data, smoothed_tb, smoothed_db, 2)

    test_tu_tr_count = test_uni_preds.count(0)
    test_tu_de_count = test_uni_preds.count(1)
    test_tb_tr_count = test_uni_preds.count(0)
    test_tb_de_count = test_uni_preds.count(1)

########################################################################
# REPORTING ************************************************************
    print('\n... TRAINED LANGUAGE MODEL REPORTING ...')

# Unknown Handling data
    print('\nUNKNOWN HANDLING')
    print('Truthful unigram UNK count:  ' + str(updated_tr_unigrams['<UNK>']))
    print('Truthful bigram UNK count:   ' + str(updated_tr_bigrams[('<UNK>', '<UNK>')]))
    print('Deceptive unigram UNK count: ' + str(updated_de_unigrams['<UNK>']))
    print('Deceptive bigram UNK count:  ' + str(updated_de_bigrams[('<UNK>', '<UNK>')]))

# Probability Data
    print('\nPROBABILITIES')
    print(f'Truthful uni probability sum = {sum(probs_tu.values())}')
    print(f'Smooth truthful bi probability sum = {sum(smoothed_tb.values())}')
    print(f'Deceptive uni probability sum = {sum(probs_du.values())}')
    print(f'Smooth deceptive bi probability sum = {sum(smoothed_db.values())}')
    print(f'\nMax truthful unigram prob = {max(probs_tu)} {max(probs_tu.values())}')
    print(f'Max truthful bigram prob = {max(smoothed_tb)} {max(smoothed_tb.values())}')    
    print(f'\nMax deceptive unigram prob = {max(probs_du)} {max(probs_du.values())}')
    print(f'Max deceptive bigram prob = {max(smoothed_db)} {max(smoothed_db.values())}')
    print('\nTruthful unigram UNK prob:  ' + str(probs_tu['<UNK>']))
    print('Truthful bigram UNK prob:   ' + str(smoothed_tb[('<UNK>', '<UNK>')]))
    print('Deceptive unigram UNK prob: ' + str(probs_du['<UNK>']))
    print('Deceptive bigram UNK prob:  ' + str(smoothed_db[('<UNK>', '<UNK>')]))

    # Validation Data Classification
    print('\n\n... VALIDATION DATA CLASSIFICATON ...')
    print(f'\nVALIDATION TRUTHFUL PREDICTIONS ({len(val_tr_data)})')
    print(f'\nUnigrams:\ttruthful = {tu_tr_count}\tdeceptive = {tu_de_count}')
    print(f'Bigrams:\ttruthful = {tb_tr_count}\tdeceptive = {tb_de_count}')
    print(f'\nVAL TRUTHFUL UNIGRAM ACCURACY = {pp.prediction_accuracy(val_tr_uni_preds, val_tr_labels)}')
    print(f'VAL TRUTHFUL BIGRAM ACCURACY = {pp.prediction_accuracy(val_tr_bi_preds, val_tr_labels)}')

    print(f'\nVALIDATION DECEPTIVE PREDICTIONS ({len(val_de_data)})')
    print(f'\nUnigrams:\ttruthful = {du_tr_count}\tdeceptive = {du_de_count}')
    print(f'Bigrams:\ttruthful = {db_tr_count}\tdeceptive = {db_de_count}')
    print(f'\nVAL DECEPTIVE UNIGRAM ACCURACY = {pp.prediction_accuracy(val_de_uni_preds, val_de_labels)}')
    print(f'VAL DECEPTIVE BIGRAM ACCURACY = {pp.prediction_accuracy(val_de_bi_preds, val_de_labels)}')

    # Test Data Classification
    print(f'\n\n... TEST DATA CLASSIFICATION ...')
    print(f'\nTEST PREDICTIONS ({len(test_labels)})')
    print(f'\nUnigrams:\ttruthful = {test_tu_tr_count}\tdeceptive = {test_tu_de_count}')
    print(f'Bigrams:\ttruthful = {test_tb_tr_count}\tdeceptive = {test_tb_de_count}')
    print(f'\nTEST UNIGRAM ACCURACY = {pp.prediction_accuracy(test_uni_preds, test_labels)}')
    print(f'TEST BIGRAM ACCURACY = {pp.prediction_accuracy(test_bi_preds, test_labels)}')
    print(f'\nunigram predictions\n{test_uni_preds}')
    print(f'\nbigram predictions\n{test_bi_preds}')
    