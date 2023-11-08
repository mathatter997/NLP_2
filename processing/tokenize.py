from math import log2, log10
from typing import List
import string
import numpy as np

class Preprocessor:
    def __init__(self, documents : List[str]):
        self.w_freqs = {}
        self.documents = []
        self.sdocuments = []
        split_docs = [[s.strip() for s in x.split('.')] for x in documents]
        for document in split_docs:
            _document = []
            _sdocument = []
            for sentence in document:
                _sentence = []
                sentence = sentence.translate(str.maketrans('', '', string.punctuation)).strip().lower()
                tokens = sentence.split(' ')
                for token in tokens:
                    if token in self.w_freqs:
                        self.w_freqs[token] += 1
                    else:
                        self.w_freqs[token] = 1
                    _sentence.append(token)
                _document.append(_sentence)
                _sdocument.append(sentence)
            self.documents.append(_document)
            self.sdocuments.append(_sdocument)

        self.lex = {word for word in self.w_freqs}
        self.word2idx= {word : i for i, word in enumerate(self.lex)}
    
    def initPPMI(self, window : int =4, alpha : float =0.75):
        self.c_freqs = {}
        self.wc_freqs = {}
        self.alpha = alpha
        for document in self.documents:
            for sentence in document:
                for i, token in enumerate(sentence):
                    context = ''.join(sentence[max(0, i - window):i]) \
                            + ''.join(sentence[i + 1 : min(i + window + 1, len(sentence))])
                    if context in self.c_freqs:
                        self.c_freqs[context] += 1
                    else:
                        self.c_freqs[context] = 1
                    
                    if (token, context) in self.wc_freqs:
                        self.wc_freqs[(token, context)] += 1
                    else:
                        self.wc_freqs[(token, context)] = 1
                    
        self.context_denom = sum(self.c_freqs[context] ** self.alpha for context in self.c_freqs)
        self.denom = sum(self.wc_freqs[wc] for wc in self.wc_freqs)
        self.ppmis = []
        def ppmi(word, context):
            pmi = log2(self.wc_freqs[(word, context)] / (self.w_freqs[word] * 
                        self.c_freqs[context] ** self.alpha  / self.context_denom))
            return max(pmi, 0)
        
        for document in self.documents:
            scores = [0 for _ in range(len(document))]
            for i, sentence in enumerate(document):
                for i, token in enumerate(sentence):
                    context = ''.join(sentence[max(0, i - window):i]) \
                                + ''.join(sentence[i + 1 : min(i + window + 1, len(sentence))])
                    score += ppmi(token, context)
                scores[i] = score
            self.ppmis.append(scores)

    # term frequency - inverse document frequency
    # normalize averages score over the number of words per sentence
    #NOTE: for tf-idf the "documents" are in fact the sentences in each document element
    #NOTE: the tf-idf scored for sentences is computed independently for each document in self._documents
    def initTFIDF(self, normalize=False):
        self.tf_idf = []
        for document in self._documents:
            tf = {}
            df = {} 
            for sentence in document:
                for token in sentence:
                    if token in tf:
                         tf[token] += 1
                    else:
                        tf[token] = 1
                for token in set(sentence):
                    if token in df:
                        df[token] += 1
                    else:
                        df[token] = 1
            scores = [0 for _ in range(len(document))]
            for i, sentence in enumerate(document):
                score = sum(log10(tf[token] + 1) * log10(len(document) / df[token]) for token in sentence)
                if normalize:
                    score /= len(sentence)
                scores[i] = score
            self.tf_idf.append(scores)
    
    def getPPMI(self, words : List[str], contexts : List[str]) -> List[float]:
        def ppmi(word, context):
            if (word, context) not in self.wc_freqs:
                return 0
            if word not in words or context not in contexts:
                return -1 
            pmi = log2(self.wc_freqs[(word, context)] / (self.w_freqs[word] * 
                        self.c_freqs[context] ** self.alpha  / self.context_denom))
            return max(pmi, 0)
        
        scores = [ppmi(words[i], contexts[i]) for i in range(len(words))]

        return scores
    

