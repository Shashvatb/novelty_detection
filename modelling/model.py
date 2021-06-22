from gensim.models import Word2Vec
import numpy as np


class MemNet(object):
    def __init__(self):
        self.I = Word2Vec(vector_size=100, window=3, min_count=1, workers=-1)

    def input_layer_train(self, x):
        x = [i.split() for i in x]
        self.I.build_vocab(x)
        total_examples = self.I.corpus_count
        self.I.train(x, total_examples=total_examples, epochs=self.I.epochs)
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
                y_hat.append(np.mean(sentence, axis=1))
        elif type(x) == str:
            words = x.split()
            sentence = []
            for j in words:
                try:
                    sentence.append(self.I.wv[j])
                except KeyError:
                    pass
            y_hat.append(np.mean(sentence, axis=1))
        else:
            print(type(x))
        return y_hat
