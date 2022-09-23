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

    # truthful preprocessing
    tr_unigrams_dict = lm.get_ngrams(tr_train_data, 1)
    tr_bigrams_dict = lm.get_ngrams(tr_train_data, 2)

    # deceptive preprocessing
    de_unigrams_dict = lm.get_ngrams(de_train_data, 1)
    de_bigrams_dict = lm.get_ngrams(de_train_data, 2)


    # UNKNOWN WORD HANDLING
    updated_tr_unigrams = lm.get_unknown_ngrams(tr_unigrams_dict, 1, 15)
    updated_tr_bigrams = lm.get_unknown_ngrams(tr_bigrams_dict, 2, 0)
    updated_de_unigrams = lm.get_unknown_ngrams(de_unigrams_dict, 1, 30)
    updated_de_bigrams = lm.get_unknown_ngrams(de_bigrams_dict, 2, 0)
    # print('Truthful unigram UNK count:  ' + str(updated_tr_unigrams['<UNK>']))
    # print('Truthful bigram UNK count:   ' + str(updated_tr_bigrams[('<UNK>', '<UNK>')]))
    # print('Deceptive unigram UNK count: ' + str(updated_de_unigrams['<UNK>']))
    # print('Deceptive bigram UNK count:  ' + str(updated_de_bigrams[('<UNK>', '<UNK>')])) 
    

    # SMOOTHING
    smoothed_tu = lm.smooth_unigrams(updated_tr_unigrams, 1)
    smoothed_tb = lm.smooth_bigrams(updated_tr_unigrams, updated_tr_bigrams, 1)
    smoothed_du = lm.smooth_unigrams(updated_de_unigrams, 1)
    smoothed_db = lm.smooth_bigrams(updated_de_unigrams, updated_de_bigrams, 1)
    
    print('\n... TRAINED MODEL REPORT ...')
    print(f'smooth truthful uni probability sum = {sum(smoothed_tu.values())}')
    print(f'smooth truthful bi probability sum = {sum(smoothed_tb.values())}')
    print(f'smooth deceptive uni probability sum = {sum(smoothed_du.values())}')
    print(f'smooth deceptive bi probability sum = {sum(smoothed_db.values())}')
    print(f'\nmax truthful unigram prob = {max(smoothed_tu)} {max(smoothed_tu.values())}')
    print(f'max truthful bigram prob = {max(smoothed_tb)} {max(smoothed_tb.values())}')    
    print(f'\nmax deceptive unigram prob = {max(smoothed_du)} {max(smoothed_du.values())}')
    print(f'max deceptive bigram prob = {max(smoothed_db)} {max(smoothed_db.values())}')
    print('\nTruthful unigram UNK prob:  ' + str(smoothed_tu['<UNK>']))
    print('Truthful bigram UNK prob:   ' + str(smoothed_tb[('<UNK>', '<UNK>')]))
    print('Deceptive unigram UNK prob: ' + str(smoothed_du['<UNK>']))
    print('Deceptive bigram UNK prob:  ' + str(smoothed_db[('<UNK>', '<UNK>')])) 
    
    # VALIDATION ***************************************************************************
    val_tr_data = read_from_file('A1_DATASET\\validation\\truthful.txt')
    val_de_data = read_from_file('A1_DATASET\\validation\\deceptive.txt')
    
    print('\n... VALIDATION DATA BEGINS ...')
  
    print(f'\n****** VALIDATION TRUTHFUL PREDICTIONS ({len(val_tr_data)}) ******')
    val_tr_uni_preds = pp.make_predictions(val_tr_data, smoothed_tu, smoothed_du, 1)
    val_tr_bi_preds = pp.make_predictions(val_tr_data, smoothed_tb, smoothed_db, 2)

    val_tr_labels = [0] * len(val_tr_uni_preds)
    tu_tr_count = val_tr_uni_preds.count(0)
    tu_de_count = val_tr_uni_preds.count(1)
    tb_tr_count = val_tr_bi_preds.count(0)
    tb_de_count = val_tr_bi_preds.count(1)

    print(f'\nUnigrams:\ttruthful = {tu_tr_count}\tdeceptive = {tu_de_count}')
    print(f'Bigrams:\ttruthful = {tb_tr_count}\tdeceptive = {tb_de_count}')
    print(f'\nVAL TRUTHFUL UNIGRAM ACCURACY = {pp.prediction_accuracy(val_tr_uni_preds, val_tr_labels)}')
    print(f'VAL TRUTHFUL BIGRAM ACCURACY = {pp.prediction_accuracy(val_tr_bi_preds, val_tr_labels)}')

    print(f'\n****** VALIDATION DECEPTIVE PREDICTIONS ({len(val_de_data)}) ******')
    val_de_uni_preds = pp.make_predictions(val_de_data, smoothed_tu, smoothed_du, 1)
    val_de_bi_preds = pp.make_predictions(val_de_data, smoothed_tb, smoothed_db, 2)
    
    val_de_labels = [1] * len(val_de_uni_preds)
    du_tr_count = val_de_uni_preds.count(0)
    du_de_count = val_de_uni_preds.count(1)
    db_tr_count = val_de_bi_preds.count(0)
    db_de_count = val_de_bi_preds.count(1)

    print(f'\nUnigrams:\ttruthful = {du_tr_count}\tdeceptive = {du_de_count}')
    print(f'Bigrams:\ttruthful = {db_tr_count}\tdeceptive = {db_de_count}')
    print(f'\nVAL DECEPTIVE UNIGRAM ACCURACY = {pp.prediction_accuracy(val_de_uni_preds, val_de_labels)}')
    print(f'VAL DECEPTIVE BIGRAM ACCURACY = {pp.prediction_accuracy(val_de_bi_preds, val_de_labels)}')


# TESTING **********************************************************************
    print(f'\n\n... TEST DATA BEGINS ...')
    test_data = read_from_file('A1_DATASET\\test\\test.txt')
    test_labels = labels_from_file('A1_DATASET\\test\\test_labels.txt')

    print(f'\n****** TEST PREDICTIONS ({len(test_labels)}) ******')
    test_uni_preds = pp.make_predictions(test_data, smoothed_tu, smoothed_du, 1)
    test_bi_preds = pp.make_predictions(test_data, smoothed_tb, smoothed_db, 2)

    tu_tr_count = test_uni_preds.count(0)
    tu_de_count = test_uni_preds.count(1)
    tb_tr_count = test_uni_preds.count(0)
    tb_de_count = test_uni_preds.count(1)

    print(f'\nUnigrams:\ttruthful = {tu_tr_count}\tdeceptive = {tu_de_count}')
    print(f'Bigrams:\ttruthful = {tb_tr_count}\tdeceptive = {tb_de_count}')
    print(f'\nTEST UNIGRAM ACCURACY = {pp.prediction_accuracy(test_uni_preds, test_labels)}')
    print(f'TEST BIGRAM ACCURACY = {pp.prediction_accuracy(test_bi_preds, test_labels)}')
    
    print(f'\nunigram predictions\n{test_uni_preds}')
    print(f'\nbigram predictions\n{test_bi_preds}')
