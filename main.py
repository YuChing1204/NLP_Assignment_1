import ngrams_model
import smoothing

if __name__ == "__main__":
    path = "A1_DATASET//train//truthful.txt"
    truthful = ngrams_model.read_file(path)
    prob_unigram = ngrams_model.prob_unigram(truthful)
    prob_bigram = ngrams_model.prob_bigram(truthful)

    print("truthful unigram:")
    print(max(prob_unigram, key=prob_unigram.get), prob_unigram[max(prob_unigram, key=prob_unigram.get)])
    print("truthful bigram:")
    print(max(prob_bigram, key=prob_bigram.get), prob_bigram[max(prob_bigram, key=prob_bigram.get)])

    prob_laplace_k_smooth_unigram = smoothing.prob_laplace_k_smooth_unigram(truthful, 1)
    print("truthful laplace unigram:")
    print(max(prob_laplace_k_smooth_unigram, key=prob_laplace_k_smooth_unigram.get),
          prob_laplace_k_smooth_unigram[max(prob_laplace_k_smooth_unigram, key=prob_laplace_k_smooth_unigram.get)])