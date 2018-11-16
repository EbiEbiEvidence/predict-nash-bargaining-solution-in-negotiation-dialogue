import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Embedding, GRU, TimeDistributed, Concatenate, RepeatVector, Permute, Lambda, Bidirectional, Multiply
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences

from .model import AttentionGRU

NB_SPOKEN_WORDS = 256
NB_DRAFTS = 64


def padding(x, max_length):
    if isinstance(x[0], int):
        return x[:max_length] + [0 for _ in range(max_length - len(x[:max_length]))]
    else:
        return np.r_[x[:max_length], [np.zeros(np.array(x[0]).shape) for _ in range(max_length - len(x[:max_length]))]]


class Processor(object):
    def __init__(self, word2id, nb_words=256, nb_issues=3):
        self.word2id = word2id
        self.nb_words = nb_words
        self.nb_issues = nb_issues
        super(Processor, self).__init__()
    
    def get_model(self):
        inp_words = Input((self.nb_words,), name='input_dialogue')
        emb_words = Embedding(output_dim=16,
                              input_dim=len(self.word2id),
                              input_length=self.nb_words,
                              trainable=True)(inp_words)
        rnn_words = AttentionGRU(input_shape=(None, self.nb_words, 16), gru_dim=256)(emb_words)
        rnn_words = Lambda(lambda x: K.sum(x, axis=1))(rnn_words)
        o = Dense(self.nb_issues, activation='softmax')(rnn_words)
        return Model(inputs=inp_words, outputs=o)
    
    def to_Xy(self, nego):
        str_dialogue = (" ".join([str(d) for d in nego.dialogues])).split(" ")
        X_dialogue = [self.word2id[word] for word in str_dialogue]
        X_dialogue = padding(np.array(X_dialogue, dtype='int32'), self.nb_words)
        y = np.array([nego.user_you.issue2weight[item] for item in ['book', 'hat', 'ball']])
        y = y / y.sum()
        return np.array(X_dialogue), y
