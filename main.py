import ngrams_model
import smoothing
import perplexity
import unknown

if __name__ == "__main__":
    # step1: read data
    # train data
    print("step1")
    train_truthful_path = "A1_DATASET//train//truthful.txt"
    train_truthful_tokens = ngrams_model.read_file(train_truthful_path)
    train_deceptive_path = "A1_DATASET//train//deceptive.txt"
    train_deceptive_tokens = ngrams_model.read_file(train_deceptive_path)
    # validation data
    validation_truthful_path = "A1_DATASET//validation//truthful.txt"
    validation_truthful_tokens = ngrams_model.read_file(validation_truthful_path)
    validation_deceptive_path = "A1_DATASET//validation//deceptive.txt"
    validation_deceptive_tokens = ngrams_model.read_file(validation_deceptive_path)
    # # test data
    test_path = "A1_DATASET//test//test.txt"
    test_tokens = ngrams_model.read_file(test_path)

    # step2: tokenize data
    # train_truthful_tokens = ngrams_model.tokens_process(train_truthful)
    # train_deceptive_tokens = ngrams_model.tokens_process(train_deceptive)
    # validation_truthful_tokens = ngrams_model.tokens_process(validation_truthful)
    # validation_deceptive_tokens = ngrams_model.tokens_process(validation_deceptive)
    # test_tokens = ngrams_model.tokens_process(test)

    # # step 3: add to vocabulary
    # vocabulary_truthful_uni_set = ngrams_model.build_uni_vocabulary([train_truthful_tokens, train_deceptive_tokens])
    # vocabulary_deceptive_uni_set = ngrams_model.build_uni_vocabulary([train_truthful_tokens, train_deceptive_tokens])
    # vocabulary_truthful_bi_set = ngrams_model.build_bi_vocabulary([train_truthful_tokens, train_deceptive_tokens])
    # vocabulary_deceptive_bi_set = ngrams_model.build_bi_vocabulary([train_truthful_tokens, train_deceptive_tokens])

    # step 2:
    # calculate truthful_unigram, truthful_bigram, deceptive_unigram, deceptive_bigram from train
    print("step2")
    count_unigram_truthful = ngrams_model.count_unigram(train_truthful_tokens)
    prob_unigram_truthful = ngrams_model.prob_unigram(count_unigram_truthful)
    # print(max(prob_unigram_truthful, key=prob_unigram_truthful.get), prob_unigram_truthful[max(prob_unigram_truthful, key=prob_unigram_truthful.get)])
    count_unigram_deceptive = ngrams_model.count_unigram(train_deceptive_tokens)
    prob_unigram_deceptive = ngrams_model.prob_unigram(count_unigram_deceptive)
    # print(max(prob_unigram_deceptive, key=prob_unigram_deceptive.get),prob_unigram_deceptive[max(prob_unigram_deceptive, key=prob_unigram_deceptive.get)])

    vocabulary_truthful_bi_set = ngrams_model.build_bi_vocabulary(count_unigram_truthful)
    # count_bigram_truthful = ngrams_model.count_bigram(train_truthful_tokens, vocabulary_truthful_bi_set)
    # prob_bigram_truthful = ngrams_model.prob_bigram(count_bigram_truthful, count_unigram_truthful)
    # print(count_bigram_truthful)
    # print(max(prob_bigram_truthful, key=prob_bigram_truthful.get), prob_bigram_truthful[max(prob_bigram_truthful, key=prob_bigram_truthful.get)])
    vocabulary_deceptive_bi_set = ngrams_model.build_bi_vocabulary(count_unigram_deceptive)
    # count_bigram_deceptive = ngrams_model.count_bigram(train_deceptive_tokens, vocabulary_deceptive_bi_set)
    # prob_bigram_deceptive = ngrams_model.prob_bigram(count_bigram_deceptive, count_unigram_deceptive)
    # print(max(prob_bigram_deceptive, key=prob_bigram_deceptive.get), prob_bigram_deceptive[max(prob_bigram_deceptive, key=prob_bigram_deceptive.get)])

    # # step 3:
    # # handle unknown word
    print("step3")
    count_unigram_truthful, del_keys_truthful = unknown.unknown(count_unigram_truthful, 1) # n = 1
    count_unigram_deceptive, del_keys_deceptive = unknown.unknown(count_unigram_deceptive, 1)  # n = 1

    prob_unigram_truthful = ngrams_model.prob_unigram(count_unigram_truthful)
    prob_unigram_deceptive = ngrams_model.prob_unigram(count_unigram_deceptive)

    train_truthful_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_truthful)
    train_deceptive_tokens = unknown.unknown_tokens_process(train_deceptive_tokens, del_keys_deceptive)
    # # validation_truthful_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_truthful)
    # # validation_deceptive_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_truthful)
    # # test_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_truthful)
    # # vocabulary_bi_set = ngrams_model.build_bi_vocabulary([train_truthful_tokens, train_deceptive_tokens])
    #
    # vocabulary_truthful_bi_set = ngrams_model.build_bi_vocabulary(count_unigram_truthful)
    count_bigram_truthful = ngrams_model.count_bigram(train_truthful_tokens, vocabulary_truthful_bi_set)

    # # train_truthful_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_deceptive)
    # # train_deceptive_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_deceptive)
    # # validation_truthful_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_deceptive)
    # # validation_deceptive_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_deceptive)
    # # test_tokens = unknown.unknown_tokens_process(train_truthful_tokens, del_keys_deceptive)
    # # vocabulary_bi_set = ngrams_model.build_bi_vocabulary([train_truthful_tokens, train_deceptive_tokens])
    # #
    # vocabulary_deceptive_bi_set = ngrams_model.build_bi_vocabulary(count_unigram_deceptive)
    count_bigram_deceptive = ngrams_model.count_bigram(train_deceptive_tokens, vocabulary_deceptive_bi_set)
    #
    # # step 4:
    print("step4")
    # # smoothing truthful_unigram, truthful_bigram, deceptive_unigram, deceptive_bigram from train
    # # prob_laplace_smooth_unigram_truthful = smoothing.prob_laplace_smooth_unigram(count_unigram_truthful) # laplace
    # prob_laplace_smooth_unigram_truthful = smoothing.prob_add_k_smooth_unigram(count_unigram_truthful, 1) # add k
    # # print(max(prob_laplace_smooth_unigram_truthful, key=prob_laplace_smooth_unigram_truthful.get),prob_laplace_smooth_unigram_truthful[max(prob_laplace_smooth_unigram_truthful, key=prob_laplace_smooth_unigram_truthful.get)])
    # # # prob_laplace_smooth_unigram_deceptive = smoothing.prob_laplace_smooth_unigram(count_unigram_deceptive) # laplace
    # prob_laplace_smooth_unigram_deceptive = smoothing.prob_add_k_smooth_unigram(count_unigram_deceptive, 1)  # add k
    # # print(max(prob_laplace_smooth_unigram_deceptive, key=prob_laplace_smooth_unigram_deceptive.get),prob_laplace_smooth_unigram_deceptive[max(prob_laplace_smooth_unigram_deceptive, key=prob_laplace_smooth_unigram_deceptive.get)])
    # # prob_laplace_smooth_bigram_truthful = smoothing.prob_laplace_smooth_bigram(count_unigram_truthful, count_bigram_truthful) # laplace
    prob_laplace_smooth_bigram_truthful = smoothing.prob_add_k_smooth_bigram(count_unigram_truthful, count_bigram_truthful, 1) # add k
    # # print(max(prob_laplace_smooth_bigram_truthful, key=prob_laplace_smooth_bigram_truthful.get),
    # #       prob_laplace_smooth_bigram_truthful[max(prob_laplace_smooth_bigram_truthful, key=prob_laplace_smooth_bigram_truthful.get)])
    # # prob_laplace_smooth_bigram_deceptive = smoothing.prob_laplace_smooth_bigram(count_unigram_deceptive, count_bigram_deceptive)
    prob_laplace_smooth_bigram_deceptive = smoothing.prob_add_k_smooth_bigram(count_unigram_deceptive, count_bigram_deceptive, 1)  # add k
    # # print(max(prob_laplace_smooth_bigram_deceptive, key=prob_laplace_smooth_bigram_deceptive.get),
    # #       prob_laplace_smooth_bigram_deceptive[max(prob_laplace_smooth_bigram_deceptive, key=prob_laplace_smooth_bigram_deceptive.get)])
    # #
    # #
    # # step 5: validation data truthful dataset: unigram-> truthful perplexity, deceptive perplexity,
    # # compare perplexity to calculate truthful percentage;
    # # bigram perplexity deceptive dataset: unigram perplexity, bigram perplexity
    print("step5")
    truthful_perplexity_list = []
    deceptive_perplexity_list = []
    with open(validation_truthful_path) as f:
        for line in f.readlines():
            truthful_perplexity_list.append(perplexity.cal_perplexity(prob_unigram_truthful, line))
            deceptive_perplexity_list.append(perplexity.cal_perplexity(prob_unigram_deceptive, line))

    count = 0
    for i in range(len(truthful_perplexity_list)):
        if truthful_perplexity_list[i] < deceptive_perplexity_list[i]:
            count += 1

    uni_truthful_percentage = (count / len(truthful_perplexity_list)) * 100
    print("validation_truthful_percentage_uni:", uni_truthful_percentage, "%")
    # # print("avg truthful data in truthful model perplexity:", sum(truthful_perplexity_list)/len(truthful_perplexity_list))
    # # print("avg deceptive data in truthful model perplexity:", sum(deceptive_perplexity_list) / len(deceptive_perplexity_list))
    # # print("\n")
    #
    truthful_perplexity_list = []
    deceptive_perplexity_list = []
    with open(validation_deceptive_path) as f:
        for line in f.readlines():
            truthful_perplexity_list.append(perplexity.cal_perplexity(prob_unigram_truthful, line))
            deceptive_perplexity_list.append(perplexity.cal_perplexity(prob_unigram_deceptive, line))

    count = 0
    for i in range(len(truthful_perplexity_list)):
        if truthful_perplexity_list[i] > deceptive_perplexity_list[i]:
            count += 1

    uni_deceptive_percentage = (count / len(deceptive_perplexity_list)) * 100
    print("validation_deceptive_percentage_uni:", uni_deceptive_percentage, "%")
    # # print("avg truthful data in deceptive model perplexity:", sum(truthful_perplexity_list)/len(truthful_perplexity_list))
    # # print("avg deceptive data in deceptive model perplexity:", sum(deceptive_perplexity_list) / len(deceptive_perplexity_list))
    # # print("\n")
    #
    truthful_perplexity_list = []
    deceptive_perplexity_list = []
    with open(validation_truthful_path) as f:
        for line in f.readlines():
            truthful_perplexity_list.append(perplexity.cal_bi_perplexity(prob_laplace_smooth_bigram_truthful, line))
            deceptive_perplexity_list.append(perplexity.cal_bi_perplexity(prob_laplace_smooth_bigram_deceptive, line))

    count = 0
    for i in range(len(truthful_perplexity_list)):
        if truthful_perplexity_list[i] < deceptive_perplexity_list[i]:
            count += 1

    bi_truthful_percentage = (count / len(truthful_perplexity_list)) * 100
    print("validation_truthful_percentage_bi:", bi_truthful_percentage, "%")
    # # # print("avg truthful data in truthful model perplexity:", sum(truthful_perplexity_list)/len(truthful_perplexity_list))
    # # # print("avg deceptive data in truthful model perplexity:", sum(deceptive_perplexity_list) / len(deceptive_perplexity_list))
    # # # print("\n")
    # #
    truthful_perplexity_list = []
    deceptive_perplexity_list = []
    with open(validation_deceptive_path) as f:
        for line in f.readlines():
            truthful_perplexity_list.append(perplexity.cal_bi_perplexity(prob_laplace_smooth_bigram_truthful, line))
            deceptive_perplexity_list.append(perplexity.cal_bi_perplexity(prob_laplace_smooth_bigram_deceptive, line))

    count = 0
    for i in range(len(truthful_perplexity_list)):
        if truthful_perplexity_list[i] > deceptive_perplexity_list[i]:
            count += 1

    bi_deceptive_percentage = (count / len(deceptive_perplexity_list)) * 100
    print("validation_deceptive_percentage_bi:", bi_deceptive_percentage, "%")
    # # print("avg truthful data in deceptive model perplexity:", sum(truthful_perplexity_list)/len(truthful_perplexity_list))
    # # print("avg deceptive data in deceptive model perplexity:", sum(deceptive_perplexity_list) / len(deceptive_perplexity_list))
    # # print("\n")
    #
    #
    # step 6: test data dataset

    # truthful_perplexity_list = []
    # deceptive_perplexity_list = []
    # label = []
    # result = []
    # with open(test_path) as f:
    #     for line in f.readlines():
    #         truthful_perplexity_list.append(perplexity.cal_perplexity(prob_laplace_smooth_unigram_truthful, line))
    #         deceptive_perplexity_list.append(perplexity.cal_perplexity(prob_laplace_smooth_unigram_deceptive, line))
    #
    # with open("A1_DATASET//test//test_labels.txt") as f:
    #     for line in f.readlines():
    #         label.append(int(line))
    #
    # for i in range(len(truthful_perplexity_list)):
    #     if truthful_perplexity_list[i] < deceptive_perplexity_list[i]:
    #         result.append(0)
    #     else:
    #         result.append(1)
    #
    # count = 0
    # for i in range(len(label)):
    #     if label[i] == result[i]:
    #         count += 1
    #
    # accuracy = (count / len(label)) * 100
    # print("uni_test_accuracy:", accuracy, "%")

    # truthful_perplexity_list = []
    # deceptive_perplexity_list = []
    # label = []
    # result = []
    # with open(test_path) as f:
    #     for line in f.readlines():
    #         truthful_perplexity_list.append(perplexity.cal_bi_perplexity(prob_laplace_smooth_bigram_truthful, line, prob_laplace_smooth_unigram_truthful))
    #         deceptive_perplexity_list.append(perplexity.cal_bi_perplexity(prob_laplace_smooth_bigram_deceptive, line, prob_laplace_smooth_unigram_deceptive))
    #
    # with open("A1_DATASET//test//test_labels.txt") as f:
    #     for line in f.readlines():
    #         label.append(int(line))
    #
    # for i in range(len(truthful_perplexity_list)):
    #     if truthful_perplexity_list[i] < deceptive_perplexity_list[i]:
    #         result.append(0)
    #     else:
    #         result.append(1)
    #
    # count = 0
    # for i in range(len(label)):
    #     if label[i] == result[i]:
    #         count += 1
    #
    # accuracy = (count / len(label)) * 100
    # print("bi_test_accuracy:", accuracy, "%")
