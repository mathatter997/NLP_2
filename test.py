import json
from random import sample
test_set = 'data/test.json' 
with open(test_set, 'r') as f:
    eval_data = json.load(f)

eval_articles = [article['article'] for article in eval_data]
documents = [[s.strip() for s in x.split('.')] for x in eval_articles]
sentences = [s for doc in documents for s in doc]

ds_idx = [[] for _ in documents]
counter = 0
for i, doc in enumerate(documents):
    for _ in doc:
        ds_idx[i].append(counter)
        counter += 1



# print(sub_idxs)
# for idx in sub_idxs:
#     print(sentences[idx])

# Read the CSV into a pandas data frame (df)
#   With a df you can do many things
#   most important: visualize data with Seaborn

datas = ['data/logistic_test_scores.csv', 'data/mlp_sigmoid_test_scores.csv', 'data/mlp_relu_test_scores.csv']
N = 40
step = 2
sen_scores = [[[] for _ in range(N)] for _ in range(len(datas))]
sen_scores_avg = [[0 for _ in range(N)] for _ in range(len(datas))]
for i, data in enumerate(datas):
    with open(data, 'r') as f:
        scores = [float(val) for val in f]
        for s, sentence in enumerate(sentences):
            length = len(sentence.split(' '))
            idx = min(N - 1, length // step)
            sen_scores[i][idx].append(scores[s])
    for idx in range(N):
        sen_scores_avg[i][idx] = sum(sen_scores[i][idx]) / len(sen_scores[i][idx])


print(sen_scores_avg)

        
# importing libraries
import matplotlib.pyplot as plt
 
X = [i for i in range(2 * N) if i % 2]
print(len(X), len(sen_scores_avg[0]))
# plotting first histogram
plt.plot(X, sen_scores_avg[0])
plt.plot(X, sen_scores_avg[1])
plt.plot(X, sen_scores_avg[2])
plt.title("Average Sentence Score vs. Sentence Length")
plt.xlabel('Sentence Length')
plt.ylabel('Average Score')

plt.legend(['Logistic', 'MLP Sigmoid', 'MLP ReLU'])

# Showing the plot using plt.show()
# plt.show()


sentence = "Victor Moses has returned to Chelsea after being ruled out for the season. The winger has been on loan at Stoke but damaged his hamstring in the 1-1 draw at West Ham United on Saturday and is expected to be out for six weeks. The 24-year-old has gone back to parent club Chelsea for treatment following the results of a scan."
print(sentence.count('.'))


import argparse
from evaluation.rouge_evaluator import RougeEvaluator
import json
import tqdm

args = argparse.ArgumentParser()
# args.add_argument('--pred_data', type=str, default='data/validation.json')
# args.add_argument('--eval_data', type=str, default='data/validation.json')
# args = args.parse_args()

evaluator = RougeEvaluator()

pred_files = ['data/logistic_fake_scores.csv',
              'data/mlp_sigmoid_fake_scores.csv',
              'data/mlp_relu_fake_scores.csv']

names = ['logistic', 'mlp_sigmoid', 'mlp_relu']


model_scores = [[] for _ in range(len(names))]
for i, pred_file in enumerate(pred_files):
    with open(pred_file, 'r') as f:
        pred_data = [float(score) for score in f]
        model_scores[i] = pred_data


for j, _ in enumerate(model_scores[0]):
    print('{} '.format(j + 1), end = ' & ')
    for i, _ in enumerate(names):
        end = ' & ' if i != len(names) - 1 else '\\\\'
        print('{:.3f} '.format(model_scores[i][j]), end = end)
    print()


