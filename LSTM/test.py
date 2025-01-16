import numpy as np
from lstm import LSTM

# init data + assume we have a vocabulary size of 50.
X = np.random.randn(1, 50, 5) # batch size = 1, embedding_dim = 50, seq_len = 5
one_hot_shape = (1, 50)

# init internals
n_neurons = (10, 20, 50) # idx = layer num
in_dim = 50
seq_len = 5
batch_size = 1

lstm = LSTM(
    n_neurons = n_neurons, 
    in_dim = in_dim,
    seq_len = 5,
    batch_size = 1
    )

print(lstm.forward(X).shape) # should be size 1 x 50
