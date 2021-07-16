# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class CNN_BiLSTM_Seq(BasicModule):

    def __init__(self, config):
        super(CNN_BiLSTM_Seq, self).__init__()
        self.config = config

        V = config.word_num
        D = config.word_embedding_dimension
        C = config.label_num
        Ci = 1
        Co = config.kernel_num
        Ks = config.kernel_sizes

        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers

        # CNN
        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(config.dropout)

        # BiLSTM
        self.bilstm = nn.LSTM(len(Ks) * Co, self.hidden_dim, num_layers=self.num_layers,
                              dropout=config.dropout, bidirectional=True, batch_first=True)

        L = self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        if self.config.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):

        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(2)  # (N, Ci, W, D)
        shape0, shape1 = x.shape[0], x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = x.view(shape0, shape1, x.shape[-1])

        bilstm_x = x
        bilstm_out, _ = self.bilstm(bilstm_x)
        bilstm_out = self.hidden2label1(F.tanh(bilstm_out))
        bilstm_out = self.hidden2label2(F.tanh(bilstm_out))

        # output
        logit = bilstm_out
        return logit

