import argparse
import json
import numpy as np
from random import sample
from summarizers.summarizers import LogisticSummarizer, ReLU_MLPSummarizer, Sigmoid_MLPSummarizer, RandomSummarizer, HackSummarizer

args = argparse.ArgumentParser()
args.add_argument('--train_data', type=str, default='data/train.greedy_sent.json')
args = args.parse_args()


models = [LogisticSummarizer(theta_length=2, wv_size=50), 
          Sigmoid_MLPSummarizer(theta_length=2, wv_size=50),
          ReLU_MLPSummarizer(theta_length=2, wv_size=50, relu_alpha=0.1), 
          RandomSummarizer(), 
          HackSummarizer()][0:1]

model_names = ['logistic', 'mlp_sigmoid', 'mlp_relu', 'random', 'hack'][0:1]

validation_set = 'data/validation.json'
test_set = 'data/test.json' 
eval_sets = [validation_set, test_set]
eval_names = ['validation', 'test']

for i, model in enumerate(models):

    with open(args.train_data, 'r') as f:
        train_data = json.load(f)


    N = 500 
    indices = [i for i in range(len(train_data))] # 0 to 9999
    sub_sample = sample(indices, N)
    train_articles = [train_data[i]['article'] for i in sub_sample]
    train_highligt_decisions = [train_data[i]['greedy_n_best_indices'] for i in sub_sample]
    model.train(train_articles, train_highligt_decisions, epochs=10, ratio=0.5)

    for k, data in enumerate(eval_sets):
        with open(eval_sets[k], 'r') as f:
            eval_data = json.load(f)

        eval_articles = [article['article'] for article in eval_data]
        summary = model.predict(eval_articles)
        eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summary)]
        with open('data/{}_{}_output.json'.format(model_names[i], eval_names[k]), 'w') as f:
            json.dump(eval_out_data, f)

        # if model_names[i] in {'logistic', 'mlp_sigmoid', 'mlp_relu'}:
        #     scores = model.score(eval_articles)
        #     np.savetxt('data/{}_{}_scores.csv'.format(model_names[i], eval_names[k]), scores, delimiter=',')

        articles = ['The', 'The' * 20, 'The' * 40, 
                                'The quick brown fox' * 2, 'The quick brown fox' * 4, 'The quick brown fox' * 8,
                                'The quick brown fox jumps over the lazy dog', 'The quick brown fox jumps over the lazy dog' * 2, 
                                'The quick brown fox jumps over the lazy dog' * 4]
        scores = model.score(articles)
        np.savetxt('data/{}_{}_scores.csv'.format(model_names[i], "fake"), scores, delimiter=',')
        


