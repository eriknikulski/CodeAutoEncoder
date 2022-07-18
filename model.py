import torch
import torch.nn as nn
import torch.nn.functional as F

import const


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, lang):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lang = lang
        self.device_count = const.CUDA_DEVICE_COUNT if const.CUDA_DEVICE_COUNT else 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, const.ENCODER_LAYERS,
                            bidirectional=True if const.BIDIRECTIONAL == 2 else False)

    def forward(self, input, hidden):
        embedded = self.embedding(input).transpose(0, 1)
        output = embedded.reshape(-1, int(self.batch_size / self.device_count), self.hidden_size)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(const.BIDIRECTIONAL * const.ENCODER_LAYERS, int(self.batch_size / self.device_count),
                           self.hidden_size, device=const.DEVICE), \
               torch.zeros(const.BIDIRECTIONAL * const.ENCODER_LAYERS, int(self.batch_size / self.device_count),
                           self.hidden_size, device=const.DEVICE)

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, lang):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lang = lang
        self.device_count = const.CUDA_DEVICE_COUNT if const.CUDA_DEVICE_COUNT else 1

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).reshape(1, int(self.batch_size / self.device_count), -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, int(self.batch_size / self.device_count), self.hidden_size, device=const.DEVICE)

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size


class DataParallel(nn.DataParallel):
    def __init__(self, my_methods, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mymethods = my_methods

    def __getattr__(self, name):
        if name in self._mymethods:
            return getattr(self.module, name)

        else:
            return super().__getattr__(name)
