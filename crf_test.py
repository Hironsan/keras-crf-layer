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
