import numpy as np


def tokenize(text):
    '''
    encodes the text by mapping each unique character
    to an integer and vice-versa
    '''
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])

    return encoded


def one_hot_encode(arr):
    '''
    one-hot encodes a given integer array
    '''
    n_labels = np.max(arr) + 1
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def get_batches(arr, batch_size, seq_length):
    '''
    a generator that returns batches of size
    batch_size x seq_length from arr
    '''

    # calculate total number of full batches
    batch_size_total = batch_size * seq_length
    n_batches = len(arr)//batch_size_total

    # keep only enough characters to make full batches
    arr = arr[:n_batches*batch_size_total]

    # reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]  # the features
        y = np.zeros_like(x)  # the targets, shifted by one
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]

        yield x, y
