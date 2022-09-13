from nltk import bigrams
from nltk.tokenize import word_tokenize


def read_file(path):
    text = ""
    with open(path) as f:
        for line in f.readlines():
            text += line
    return text


def tokens_process(tokens):
    new_tokens = []
    for token in tokens:
        if token != ',' and token != '.' and token != '!':
            new_tokens.append(token)

    return new_tokens


def count_unigram(data):
    unigram_dic = {}
    tokens = word_tokenize(data)
    new_tokens = tokens_process(tokens)

    for token in new_tokens:
        if token not in unigram_dic:
            unigram_dic[token] = 1
        else:
            unigram_dic[token] += 1

    return unigram_dic


def prob_unigram(data):
    unigram_dic = count_unigram(data)
    prob_unigram_dic = {}
    dic_len = len(unigram_dic)
    for unigram in unigram_dic:
        prob_unigram_dic[unigram] = unigram_dic[unigram] / dic_len

    return prob_unigram_dic


def count_bigram(data):
    bigram_dic = {}
    tokens = word_tokenize(data)
    new_tokens = tokens_process(tokens)
    bigram = bigrams(new_tokens)
    for bi in bigram:
        if bi not in bigram_dic:
            bigram_dic[bi] = 1
        else:
            bigram_dic[bi] += 1

    return bigram_dic


def prob_bigram(data):
    unigram_dic = count_unigram(data)
    bigram_dic = count_bigram(data)
    prob_bigram_dic = {}
    for bigram in bigram_dic:
        prob_bigram_dic[bigram] = bigram_dic[bigram] / unigram_dic[bigram[0]]

    return prob_bigram_dic


if __name__ == "__main__":
    path = "A1_DATASET//train//truthful.txt"
    truthful = read_file(path)
    prob_unigram = prob_unigram(truthful)
    prob_bigram = prob_bigram(truthful)

    print("truthful unigram:")
    print(max(prob_unigram, key=prob_unigram.get), prob_unigram[max(prob_unigram, key=prob_unigram.get)])
    print("truthful bigram:")
    print(max(prob_bigram, key=prob_bigram.get), prob_bigram[max(prob_bigram, key=prob_bigram.get)])


