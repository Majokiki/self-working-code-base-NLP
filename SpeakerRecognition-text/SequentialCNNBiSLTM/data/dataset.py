import torch
from torch.utils import data
from collections import Counter
from torchtext.vocab import Vocab

import random
import os
import re


class SequentialDataset(data.Dataset):
    def __init__(self, file_path, config):
        self.block_size = config.block_size
        self.overlap = config.overlap

        with open(file_path) as fin:
            file_str = fin.read()
        blocks = file_str.split('\n\n\n')
        tokens = []
        labels = []
        masks = []
        padding = ['<pad>'] * config.sentence_max_size
        for block in blocks:
            block_data = []
            for sent in block.split('\n'):
                if sent[:2] not in ['医生', '患者']:
                    continue
                truth = {'医生': 'doctor', '患者': 'patient'}[sent[:2]]
                sent = sent[3:].split('（男')[0].split('（女')[0]
                sent_pieces = re.split("\。|\ |\,|\，|\？|\?", sent)
                block_data.extend([[sent_piece, truth]
                                   for sent_piece in sent_pieces if sent_piece.strip()])
            block_tokens = []  # (n_segment, BLOCK_SIZE, max_len)
            block_labels = []  # (n_segment, BLOCK_SIZE)
            block_masks = []  # (n_segment, BLOCK_SIZE)
            num = 0
            while num * self.block_size - num * self.overlap < len(block_data):
                segment_tokens = []  # (BLOCK_SIZE, max_len)
                segment_labels = []  # (BLOCK_SIZE)
                segment_masks = []  # (BLOCK_SIZE)
                segment_data = block_data[num * self.block_size - num *
                                          self.overlap: (num+1) * self.block_size - num * self.overlap]
                num += 1
                for d in segment_data:
                    t = list(d[0])
                    t = t[:config.sentence_max_size] + \
                        padding[:max(0, config.sentence_max_size - len(t))]
                    l = d[1]
                    segment_tokens.append(t)
                    segment_labels.append(l)
                    segment_masks.append(1)
                if len(segment_data) < self.block_size:
                    pad_len = self.block_size - len(segment_data)
                    segment_tokens += [padding] * pad_len
                    segment_labels += [''] * pad_len
                    segment_masks += [0] * pad_len

                block_tokens.append(segment_tokens)
                block_labels.append(segment_labels)
                block_masks.append(segment_masks)

            tokens += block_tokens
            labels += block_labels
            masks += block_masks

        text_counter = Counter([k for i in tokens for j in i for k in j])
        label_counter = Counter([j for i in labels for j in i if j])
        self.text_vocab = Vocab(text_counter)
        self.label_vocab = Vocab(label_counter, specials=[])

        # (n_segments, BLOCK_SIZE, max_len)
        self.train_text_list = [torch.tensor(
            [self.text_vocab.lookup_indices(l2) for l2 in l1]) for l1 in tokens]
        # (n_segments, BLOCK_SIZE)
        self.label_list = [torch.tensor(
            [self.label_vocab.stoi[l2] if l2 else 100 for l2 in l1]) for l1 in labels]
        # (n_segments, BLOCK_SIZE)
        self.mask_list = [torch.tensor(mask) for mask in masks]

    def __getitem__(self, index):
        return self.train_text_list[index], self.label_list[index], self.mask_list[index]

    def __len__(self):
        return len(self.train_text_list)
