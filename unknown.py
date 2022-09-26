from nltk import bigrams


def unknown(count, n):
    count = count.copy()
    count["<unk>"] = 0
    del_keys = set()
    for n_gram in count:
        if n >= count[n_gram]:
            count["<unk>"] += 1
            del_keys.add(n_gram)

    for key in del_keys:
        del count[key]

    return count, del_keys

def unknow_bigram(tokens):
    bigram = bigrams(tokens)
    count = {}

    for n_gram in bigram:
        if n_gram not in count:
            count[n_gram] = 1
        else:
            count[n_gram] += 1

    return count
def unknown_tokens_process(tokens, del_keys):
    new_tokens = tokens
    for i in range(len(new_tokens)):
        token = new_tokens[i]
        if token in del_keys:
            new_tokens[i] = "<unk>"

    return new_tokens
