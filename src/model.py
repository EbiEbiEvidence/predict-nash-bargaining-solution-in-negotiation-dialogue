from keras.layers import Input, Dense, Embedding, GRU, TimeDistributed, Concatenate, RepeatVector, Permute, Lambda, Bidirectional, Multiply
from keras.models import Model


def AttentionGRU(input_shape, gru_dim, single=False):
    batch_size, time_steps, embedding_dim = input_shape
    i = Input(shape=(time_steps, embedding_dim, ), dtype='float32')
    g = Bidirectional(GRU(gru_dim, activation='relu', dropout=0.5, return_sequences=True))(i)
    g = Bidirectional(GRU(gru_dim, activation='relu', dropout=0.5, return_sequences=True))(g)
    g = Bidirectional(GRU(gru_dim, activation='relu', dropout=0.5, return_sequences=True))(g)
    a = Permute((2, 1))(g)
    a = Dense(time_steps, activation='softmax')(a)
    if single:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(gru_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='attention_mul')([g, a_probs])
    return Model(i, output_attention_mul, name='attention')
