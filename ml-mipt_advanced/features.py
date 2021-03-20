from collections import OrderedDict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
# import heapq

class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, corpus: np.ndarray, y = None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow


        self.bow = dict()
        for index in range(len(corpus)):
            tokens = list(corpus[index].split())
            for token in tokens:
                if token not in self.bow.keys(): self.bow[token] = 1
                else: self.bow[token] += 1

        # print(list(dict(sorted(self.bow.items(), key = lambda item: -item[1])[:self.k]).keys()) == heapq.nlargest(self.k, self.bow, key = self.bow.get))
        self.bow = list(dict(sorted(self.bow.items(),
                                    key = lambda item: -item[1])[:self.k]).keys())
      
        # raise NotImplementedError
        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
        result = np.zeros(len(self.bow))
        tokens = text.split()
        for ind, token in enumerate(self.bow):
            if token in tokens: result[ind] += tokens.count(token)
        return np.array(result, "float32")

    def transform(self, corpus: np.ndarray, y = None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(corpus[index]) for index in range(len(corpus))])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow

class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.tf_idf = None

    def fit(self, corpus: np.ndarray, y = None):
        """
        :param corpus: array of texts to be trained on
        """
        # raise NotImplementedError
        all_tokens = set(' '.join(corpus).split())
        if self.k is None: self.k = len(all_tokens)

        self.wordfreq = dict()
        for document in corpus:
            for term in document.split():
                if term not in self.wordfreq.keys(): self.wordfreq[term] = 1
                else: self.wordfreq[term] += 1

        self.wordfreq = list(dict(sorted(self.wordfreq.items(),
                                    key = lambda item: -item[1])[:self.k]).keys())
        

        self.idf = OrderedDict()
        for term in self.wordfreq:
            doc_containing_word = 0
            for document in corpus:
                if term in document.split(): doc_containing_word += 1
            self.idf[term] = np.log(len(corpus)/(1 + doc_containing_word))


        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        result = []
        tmp_text = text.split()
        for term in self.wordfreq:
            if term in tmp_text:
                n = tmp_text.count(term)
                if self.normalize: n /= len(tmp_text)
                result.append(n * self.idf[term])
            else: result.append(.0)
        
        # raise NotImplementedError
        return np.array(result, "float64")

    def transform(self, X: np.ndarray, y = None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])