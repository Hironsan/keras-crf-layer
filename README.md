# Keras-CRF-Layer
The Keras-CRF-Layer module implements a linear-chain CRF layer for learning to predict tag sequences.
This variant of the CRF is factored into unary potentials for every element in the sequence and binary potentials for every transition between output tags.

## Usage
Below is an example of the API, which learns a CRF for some random data.
The linear layer in the example can be replaced by any neural network.

```python
import numpy as np
from keras.layers import Embedding, Input
from keras.models import Model

from crf import CRFLayer

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

# Build an example model.
word_ids = Input(batch_shape=(batch_size, maxlen), dtype='int32')
sequence_lengths = Input(batch_shape=[batch_size, 1], dtype='int32')

word_embeddings = Embedding(vocab_size, n_classes)(word_ids)
crf = CRFLayer()
pred = crf(inputs=[word_embeddings, sequence_lengths])
model = Model(inputs=[word_ids, sequence_lengths], outputs=[pred])
model.compile(loss=crf.loss, optimizer='sgd')

# Train first 1 batch.
model.train_on_batch([x, s], y)

# Save the model
model.save('model.h5')
```

### Model loading                                                                                                       
When you want to load a saved model that has a crf output, then loading
the model with 'keras.models.load_model' won't work properly because
the reference of the loss function to the transition parameters is lost. To
fix this, you need to use the parameter 'custom_objects' as follows: 

```python
from keras.models import load_model

from crf import create_custom_objects

model = load_model('model.h5', custom_objects=create_custom_objects())
```
