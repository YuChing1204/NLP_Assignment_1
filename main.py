import ngrams_model
import smoothing
import perplexity

if __name__ == "__main__":
    path = "A1_DATASET//train//truthful.txt"
    truthful = ngrams_model.read_file(path)
    prob_unigram = ngrams_model.prob_unigram(truthful)
    prob_bigram = ngrams_model.prob_bigram(truthful)

    print("truthful unigram:")
    print(max(prob_unigram, key=prob_unigram.get), prob_unigram[max(prob_unigram, key=prob_unigram.get)])
    print("truthful bigram:")
    print(max(prob_bigram, key=prob_bigram.get), prob_bigram[max(prob_bigram, key=prob_bigram.get)])

    prob_laplace_smooth_unigram = smoothing.prob_laplace_smooth_unigram(truthful)
    print("truthful laplace unigram:")
    print(max(prob_laplace_smooth_unigram, key=prob_laplace_smooth_unigram.get),
          prob_laplace_smooth_unigram[max(prob_laplace_smooth_unigram, key=prob_laplace_smooth_unigram.get)])

    prob_laplace_smooth_bigram = smoothing.prob_laplace_smooth_bigram(truthful)
    print("truthful laplace bigram:")
    print(max(prob_laplace_smooth_bigram, key=prob_laplace_smooth_bigram.get),
          prob_laplace_smooth_bigram[max(prob_laplace_smooth_bigram, key=prob_laplace_smooth_bigram.get)])

    prob_add_k_smooth_unigram = smoothing.prob_add_k_smooth_unigram(truthful, 2)
    print("truthful add-k unigram:")
    print(max(prob_add_k_smooth_unigram, key=prob_add_k_smooth_unigram.get),
          prob_add_k_smooth_unigram[max(prob_add_k_smooth_unigram, key=prob_add_k_smooth_unigram.get)])

    prob_add_k_smooth_bigram = smoothing.prob_add_k_smooth_bigram(truthful, 2)
    print("truthful add-k bigram:")
    print(max(prob_add_k_smooth_bigram, key=prob_add_k_smooth_bigram.get),
          prob_add_k_smooth_bigram[max(prob_add_k_smooth_bigram, key=prob_add_k_smooth_bigram.get)])

    print("truthful unigram perplexity")
    print(perplexity.cal_perplexity(prob_unigram))

    print("truthful bigram perplexity")
    print(perplexity.cal_perplexity(prob_bigram))


