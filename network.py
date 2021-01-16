import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=512, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.lstm = nn.LSTM(input_size=len(self.chars), hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        # stack up LSTM outputs
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        '''
        Initializes hidden state
        '''
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (
                weight.new(self.n_layers, batch_size,
                           self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size,
                           self.n_hidden).zero_().cuda()
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size,
                           self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size,
                           self.n_hidden).zero_()
            )

        return hidden
