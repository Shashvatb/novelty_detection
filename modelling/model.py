from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os


class MemNet(object):
    def __init__(self, path, train=None):
        self.path = path
        if not train:
            with open(os.path.join(path, 'vectorizer.pkl'), 'rb') as f:
                self.I = pickle.load(f)
            with open(os.path.join(path, 'memory.pkl'), 'rb') as f:
                self.M = pickle.load(f)

        else:
            self.M = None
            self.I = Word2Vec(vector_size=100, window=3, min_count=1, workers=-1)
        self.G = cosine_similarity

    def input_layer_train(self, x):
        x = [i.split() for i in x]
        self.I.build_vocab(x)
        total_examples = self.I.corpus_count
        self.I.train(x, total_examples=total_examples, epochs=self.I.epochs)
        with open(os.path.join(self.path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.I, f)
        print('model trained')

    def input_layer(self, x):
        y_hat = []
        if type(x) == list or type(x) == np.ndarray:

            for i in x:
                sentence = []
                words = i.split()
                for j in words:
                    try:
                        sentence.append(self.I.wv[j])
                    except KeyError:
                        pass
                y_hat.append(np.mean(sentence, axis=0))
        elif type(x) == str:
            words = x.split()
            sentence = []
            for j in words:
                try:
                    sentence.append(self.I.wv[j])
                except KeyError:
                    pass
            y_hat.append(np.mean(sentence, axis=0))
        else:
            print(type(x))
        return y_hat

    def memory_unit_train(self, x):
        vectors = self.input_layer(x)

        assert len(vectors[0]) == len(vectors[1])

        with open(os.path.join(self.path, 'memory.pkl'), 'wb') as f:
            pickle.dump(vectors, f)

    def memory_unit(self, x):
        if type(x) == list or type(x) == np.ndarray or type(x) == str:
            similarities = []
            if type(x) == str:
                x = [x]
            for i in x:
                best = 0.0
                for j in self.M:
                    temp = self.G(j, i)
                    if temp > best:
                        best = temp
                similarities.append(best)
            return similarities
        else:
            print(type(x))
            return None

    def inf(self, x, threshold):
        vals = []
        for i in x:
            if i >= threshold:
                vals.append(True)
            else:
                vals.append(False)
        return vals
