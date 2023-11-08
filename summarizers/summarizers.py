from tqdm import tqdm
import random
import numpy as np
from processing.tokenize import Preprocessor
from processing.model import NeuralNet
from processing.layers import *
from encoder.pca import PCAEncoder
from math import ceil
from gensim.models import Word2Vec

class NeuralSummarizer:    
    def __init__(self, theta_length:int =50, wv_size:int =1000, wv_window:int =5, ppmi_window:int =4, alpha:float =0.75, normalize=False):
        self.theta_length = theta_length
        self.pca_dim = theta_length - 2
        self.wv_size = wv_size
        self.wv_window = wv_window
        self.ppmi_window = ppmi_window
        self.alpha = alpha
        self.normalize = normalize

        self.model = None
    
    # return emedding of each sentence: [word2vec centroid, tdidf score, ppmi score, sentence position]
    def preprocess(self, documents, init=False):
        preprocessor = Preprocessor(documents)
        if init:
            self.preprocessor = preprocessor
            self.word2vec = Word2Vec(sentences=self.preprocessor.sdocuments, 
                                    vector_size=self.wv_size, window=self.wv_window, min_count=1, workers=4)
            # self.preprocessor.initPPMI(window=self.ppmi_window, alpha=self.alpha)
            # self.preprocessor.initTFIDF(normalize=self.normalize)
            wvs = np.array([self.word2vec.wv[i] for i in range(len(self.word2vec.wv))])
            self.pca = PCAEncoder(wvs, ndim=self.pca_dim)

        X = []
        for d in tqdm(range(len(preprocessor.documents)), desc="Preprocessing documents"):
            document = preprocessor.documents[d]
            embeddings = []
            for s, sentence in enumerate(document):
                centroid = np.zeros(self.pca_dim)
                if len(sentence) > 0:
                    vectors = np.zeros(shape=(len(sentence), self.wv_size))
                    for i, word in enumerate(sentence):
                        if word in self.word2vec.wv:
                            vectors[i] = self.word2vec.wv[word]      
                    projs = self.pca.project(vectors)
                    centroid = np.sum(projs, axis=0) / len(sentence)

                # tfidf = self.preprocessor.tf_idf[d][s]
                # ppmi = self.preprocessor.ppmis[d][s]
                length = len(sentence)

                embedding = np.concatenate((centroid, [s + 1], [length]),)
                embeddings.append(embedding.astype('float64'))
            X.append(embeddings)
        return X

    def train(self, X, y, epochs:int =1, eta:float =0.01, lamb:float = 0.001, batch_size=64, ratio=0.5):
        '''
            ratio: portion of postive samples to use in training data. default is 50-50 split
        '''
        X = self.preprocess(X, init=True)
        for article, decisions in tqdm(zip(X, y), desc="Validating data shape", total=len(X)):
            assert len(article) == len(decisions), "Article and decisions must have the same length"

        X = np.array([sentence for document in X for sentence in document])
        y = np.array([decision for document in y for decision in document])

        # number of selected sentences is sparse, need to randomly oversample to prevent model from fitting to 0
        ind = np.arange(y.size)
        ind0 = ind[y[ind] == 0]
        ind1 = ind[y[ind] == 1]
        
        # add minimum # of samples to balance the datasets (might be off by 1)
        if ind1.size / (y.size) > ratio:
            alpha = (1 - ratio) / ratio 
            num_samples = int(alpha * ind1.size)  - ind0.size
            samples = np.random.choice(ind0, num_samples)
            X = X + X[samples]
            y = y + y[samples]

        elif ind1.size / (y.size) < ratio:
            alpha = ratio / (1 - ratio)
            num_samples = int(alpha * ind0.size)  - ind1.size
            samples = np.random.choice(ind1, num_samples)
            X = np.concatenate((X, X[samples]))
            y = np.concatenate((y, y[samples]))

        for epoch in tqdm(range(epochs)):
            p = np.random.permutation(len(X))
            X = X[p]
            y = y[p]
            for batch in range(ceil(len(X)/batch_size)):
                i = batch * batch_size
                j = min((batch + 1) * batch_size, len(X))
                self.model.batch_update(X[i:j], y[i:j], eta=eta, lamb=lamb)

    def predict(self, X, k=3):
        """
        X: list of documents
        """
        
        docs = [[s.strip() for s in x.split('.')] for x in X]
        X_emb = self.preprocess(X)
        
        for i, article in tqdm(enumerate(docs), desc="Running extractive summarizer"):
            sentences = X_emb[i]
            sentence_scores = self.model.batch_predict(sentences)

            # Pick the top k sentences as summary.
            # Note that this is just one option for choosing sentences.
            top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
            top_sentences = [article[i] for i in top_k_idxs]
            summary = ' . '.join(top_sentences)
            
            yield summary
    
    def score(self, X, k=3):
        """
        X: list of documents
        """
        
        docs = [[s.strip() for s in x.split('.')] for x in X]
        X_emb = self.preprocess(X)
        sentence_scores = []
        for i, article in tqdm(enumerate(docs), desc="Running extractive summarizer"):
            sentences = X_emb[i]
            sentence_scores += self.model.batch_predict(sentences)
            
        return np.array(sentence_scores)
    

class LogisticSummarizer(NeuralSummarizer):
    
    def __init__(self, theta_length:int =100, wv_size:int =1000, wv_window:int =5, ppmi_window:int =4, alpha:float =0.75, normalize=False):
        super().__init__(theta_length, wv_size, wv_window, ppmi_window, alpha, normalize)
        self.model = NeuralNet([SigmoidLayer(input=theta_length, output=1)])
    

class Sigmoid_MLPSummarizer(NeuralSummarizer):
    def __init__(self, theta_length:int =100, wv_size:int =1000, wv_window:int =5, ppmi_window:int =4, alpha:float =0.75, normalize=False, size=32):
        super().__init__(theta_length, wv_size, wv_window, ppmi_window, alpha, normalize)
        self.model = NeuralNet([SigmoidLayer(input=self.theta_length, output=size),
                                SigmoidLayer(input=size, output=size),
                                SigmoidLayer(input=size, output=1)])

class ReLU_MLPSummarizer(NeuralSummarizer):
    def __init__(self, theta_length:int =100, wv_size:int =1000, wv_window:int =5, ppmi_window:int =4, alpha:float =0.75, normalize=False, size=32, relu_alpha=1):
        super().__init__(theta_length, wv_size, wv_window, ppmi_window, alpha, normalize)
        self.model = NeuralNet([ReLULayer(input=self.theta_length, output=size, alpha=relu_alpha),
                                ReLULayer(input=size, output=size, alpha=relu_alpha),
                                SigmoidLayer(input=size, output=1)])


class RandomSummarizer:

    def preprocess(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        return split_articles

    def train(self, X, y, epochs=1, ratio=0.5):
        pass

    def predict(self, X, k=3):
        """
        X: list of documents
        """
        # X -> list of list of sentences
        X = self.preprocess(X) 
        for article in tqdm(X, desc="Running extractive summarizer"):
            # Randomly assign a score to each sentence. 
            # This is just a placeholder for your actual model.
            sentence_scores = [random.random() for _ in article]
            # Pick the top k sentences as summary.
            # Note that this is just one option for choosing sentences.
            top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
            top_sentences = [article[i] for i in top_k_idxs]
            summary = ' . '.join(top_sentences)
            
            yield summary

class HackSummarizer:
    def preprocess(self, X):
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        return split_articles

    def train(self, X, y, epochs=1, ratio=0.5):
       pass

    def predict(self, X, k=3):
        """
        X: list of documents
        """
        # X -> list of list of sentences
        X = self.preprocess(X) 
        for article in tqdm(X, desc="Running extractive summarizer"):
            # Randomly assign a score to each sentence. 
            # This is just a placeholder for your actual model.
            # Pick the top k sentences as summary.
            # Note that this is just one option for choosing sentences.
            summary = 'The time to be of that .'
            yield summary

