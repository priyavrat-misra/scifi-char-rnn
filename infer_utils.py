import torch
import torch.nn.functional as F
import numpy as np
from utils import one_hot_encode

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict(model, char, h=None, top_k=None):
    '''
    Given a character predicts the next character.
    Returns the predicted character and the hidden state.
    '''
    x = np.array([[model.char2int[char]]])
    x = one_hot_encode(x, len(model.chars))
    inputs = torch.from_numpy(x).to(device)

    with torch.no_grad():
        out, h = model(inputs, h)

    p = F.softmax(out, dim=1).data
    if device == 'cuda':
        p = p.cpu()

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(model.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next char with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    return model.int2char[char], h


def sample(model, size, prime='The', top_k=None):
    model.to(device)

    # run through the prime chars
    chars = [ch for ch in prime]
    with torch.no_grad():
        h = model.init_hidden(1)
        for ch in prime:
            char, h = predict(model, ch, h, top_k=top_k)

        chars.append(char)

        # pass previous char and get a new one
        for ii in range(size):
            char, h = predict(model, chars[-1], h, top_k=top_k)
            chars.append(char)

    return ''.join(chars)
