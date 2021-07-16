# -*- coding: utf-8 -*-
import os
import re
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from config import Config
from model import CNN_BiLSTM_Seq
from model import train
from data import SequentialDataset
from torch.utils.data import random_split, Subset
import argparse

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--block_size', type=int, default=20)
parser.add_argument('--overlap', type=int, default=2)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--label_num', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_model_dir', type=str, default='saved_model', help='model saving directory')
parser.add_argument('--load_model_dir', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('--predict_file_path', type=str, default=None, help='predict the sentences in file')
args = parser.parse_args()


torch.manual_seed(args.seed)

device = torch.device(F"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Create the configuration
config = Config(sentence_max_size=25,
                batch_size=args.batch_size,
                block_size=args.block_size,
                overlap=args.overlap,
                word_num=11000,
                label_num=args.label_num,
                learning_rate=args.lr,
                cuda=args.gpu,
                epoch=args.epoch)

model = CNN_BiLSTM_Seq(config)
if args.load_model_dir is not None:
    model_path = os.path.join(args.load_model_dir, 'model.ckpt')
    print('\nLoading model from {}...'.format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)


# train or predict
# predict whole file
if args.train:
    data_set = SequentialDataset(file_path='data/train/chunyu-dialog-500w.train', config=config)
    val_len = int(len(data_set) // (1 / 0.1))
    train_len = len(data_set) - val_len
    indices = list(range(len(data_set)))
    train_indices, val_indices = indices[:train_len], indices[train_len:]
    training_set = Subset(data_set, train_indices)
    validation_set = Subset(data_set, val_indices)
    #training_set, validation_set = random_split(data_set, [train_len, val_len])

    training_iter = data.DataLoader(dataset=training_set,
                                    batch_size=config.batch_size,
                                    num_workers=1,
                                    drop_last=True)

    validation_iter = data.DataLoader(dataset=validation_set,
                                    batch_size=config.batch_size,
                                    num_workers=1,
                                    drop_last=True)
    # Train the model
    print()
    print(config.__dict__)
    try:
        save_model_dir = args.save_model_dir
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)
        torch.save(data_set.text_vocab, os.path.join(args.save_model_dir, 'text_vocab.pth'))
        torch.save(data_set.label_vocab, os.path.join(args.save_model_dir, 'label_vocab.pth'))
        train.train(model, config, training_iter, validation_iter, device, save_model_dir)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')


if args.predict_file_path is not None:
    total = 0
    count = 0
    path = args.predict_file_path
    text_vocab_path = os.path.join(args.load_model_dir, 'text_vocab.pth')
    label_vocab_path = os.path.join(args.load_model_dir, 'label_vocab.pth')
    text_vocab = torch.load(text_vocab_path)
    label_vocab = torch.load(label_vocab_path)
    print('Label\tInput')
    with open(path) as fin:
        seqs = []
        for line in fin:
            line = line.strip()
            truth = line[:2]
            line = line[3:]
            if not line:
                if seqs:
                    labels = train.predict([s[1] for s in seqs], 
                                            model, 
                                            config,
                                            text_vocab, 
                                            label_vocab, 
                                            device)
                    for i, ((truth, line), label) in enumerate(zip(seqs, labels)):
                        print(F'{truth} \t {label} \t {line}')
                        total += 1
                        if truth == {'doctor':'医生', 'patient':'患者'}[label]:
                            count += 1
                    print('\n\n')
                    seqs = []
            else:
                seqs.append([truth, line])
                #seqs.extend([[truth, p] for p in re.split("\。|\ |\,|\，|\？|\?", line) if p])
    print(F"Accuracy \t {count/total}")
