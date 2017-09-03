import unittest

import numpy as np
from keras.layers import Embedding, Input
from keras.models import Model, load_model

from crf import CRFLayer, create_custom_objects


class LayerTest(unittest.TestCase):

    def setUp(self):
        self.filename = 'test.h5'

    def test_crf_layer(self):

        # Hyperparameter settings.
        vocab_size = 20
        n_classes = 11
        batch_size = 2
        maxlen = 2

        # Random features.
        x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))

        # Random tag indices representing the gold sequence.
        y = np.random.randint(n_classes, size=(batch_size, maxlen))
        y = np.eye(n_classes)[y]

        # All sequences in this example have the same length, but they can be variable in a real model.
        s = np.asarray([maxlen] * batch_size, dtype='int32')
        print(x)
        print(y)
        print(s)

        # Build a model
        word_ids = Input(batch_shape=(batch_size, maxlen), dtype='int32')
        word_embeddings = Embedding(vocab_size, n_classes)(word_ids)
        sequence_lengths = Input(batch_shape=[batch_size, 1], dtype='int32')
        crf = CRFLayer()
        pred = crf([word_embeddings, sequence_lengths])
        model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
        model.compile(loss=crf.loss, optimizer='sgd')

        # Train first 1 batch
        model.train_on_batch([x, s], y)

        # Save the model
        model.save(self.filename)

    def test_load_model(self):
        model = load_model(self.filename, custom_objects=create_custom_objects())

    def test_bilstm_crf(self):

        # imports
        import keras.backend as K
        from keras.layers import Lambda, LSTM, Dropout, Dense, Bidirectional
        from keras.layers.merge import Concatenate

        # Hyperparameter settings.
        vocab_size = 10000
        word_embedding_size = 100
        char_vocab_size = 80
        char_embedding_size = 25
        num_char_lstm_units = 25
        num_word_lstm_units = 100
        dropout = 0.5
        ntags = 10

        # Build bidirectional lstm-crf model
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        word_embeddings = Embedding(input_dim=vocab_size,
                                    output_dim=word_embedding_size,
                                    mask_zero=True)(word_ids)

        """
        # build character based word embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32')
        char_embeddings = Embedding(input_dim=char_vocab_size,
                                    output_dim=char_embedding_size,
                                    mask_zero=True
                                    )(char_ids)
        s = K.shape(char_embeddings)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], char_embedding_size)))(char_embeddings)


        fwd_state = LSTM(num_char_lstm_units, return_state=True)(char_embeddings)[-2]
        bwd_state = LSTM(num_char_lstm_units, return_state=True, go_backwards=True)(char_embeddings)[-2]
        char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])

        char_embeddings = Bidirectional(LSTM(units=num_char_lstm_units, return_sequences=True))(char_embeddings)
        # shape = (batch size, max sentence length, char hidden size)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * num_char_lstm_units]))(char_embeddings)

        # combine characters and word
        x = Concatenate(axis=-1)([word_embeddings, char_embeddings])
        x = Dropout(dropout)(x)
        """

        x = Bidirectional(LSTM(units=num_word_lstm_units, return_sequences=True))(word_embeddings)
        x = Dropout(dropout)(x)
        x = Dense(ntags)(x)

        sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')

        crf = CRFLayer()
        pred = crf([x, sequence_lengths])

        model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
        model.compile(loss=crf.loss, optimizer='sgd')

        model.save(self.filename)
