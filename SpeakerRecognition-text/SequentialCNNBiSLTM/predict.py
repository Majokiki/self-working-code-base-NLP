import re
import os
import torch
from config import Config
from model import CNN_BiLSTM_Seq
from model import train
from data import SequentialDataset

import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--asr_file_path', type=str, default=None, help='predict the sentences in file')
parser.add_argument('--load_model_dir', type=str, default='best_model/', help='model directory')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(0)

device = torch.device(F"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(args.load_model_dir, 'model.ckpt')
text_vocab_path = os.path.join(args.load_model_dir, 'text_vocab.pth')
label_vocab_path = os.path.join(args.load_model_dir, 'label_vocab.pth')

# Create the configuration
config = Config(sentence_max_size=25,
                batch_size=16,
                word_num=11000,
                label_num=2,
                cuda=args.gpu)

model = CNN_BiLSTM_Seq(config)
print('\nLoading model from {}...'.format(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)

print('\nLoading vocab file from {}...'.format(text_vocab_path))
text_vocab = torch.load(text_vocab_path)
label_vocab = torch.load(label_vocab_path)


def predict_and_print(sents):
    sents_pieces = [[s for s in re.split("\。|\ |\,|\，|\？|\?", sent) if s] for sent in sents]
    labels = train.predict([s for sent_pieces in sents_pieces for s in sent_pieces], 
                            model, 
                            config,
                            text_vocab, 
                            label_vocab, 
                            device)

    # 按标点符号切分， 打印结果
    print('\n\n\n')
    labels_iter = iter(copy.copy(labels))
    for i, sent_pieces in enumerate(sents_pieces):
        print('-' * 10 + str(i) + '-' * 10)
        for sent_piece in sent_pieces:
            print(F'{next(labels_iter)}   \t {sent_piece}')

    # 按原来的句子分割，打印结果
    print('\n\n\n')
    sents_labels = []
    num = 0
    for sent_pieces in sents_pieces:
        sents_labels.append(labels[num:num+len(sent_pieces)])
        num += len(sent_pieces)

    def most_common(lst):
        if not lst:
            return 'patient'
        return max(set(lst), key=lst.count)
    sents_label = [most_common([x for x in sent_labels if x != 'null']) for sent_labels in sents_labels]
    for s, l in zip(sents, sents_label):
        print(F"{l}   \t {s}")

    # 按说话人重新分割，打印结果
    print('\n\n\n')
    print('Speaker Recognition Result:')
    pieces = [sent_piece for sent_pieces in sents_pieces for sent_piece in sent_pieces]
    cur = labels[0]
    tmp_pieces = [pieces[0]]
    for label, piece in zip(labels[1:], pieces[1:]):
        if label == cur:
            tmp_pieces.append(piece)
        else:
            print(F"{cur}   \t {'，'.join(tmp_pieces) + '。'}")
            cur = label
            tmp_pieces = [piece]
    print(F"{cur}   \t {'，'.join(tmp_pieces) + '。'}")

if args.asr_file_path is not None:
    path = args.asr_file_path
    sents = []
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                if sents:
                    predict_and_print(sents)
                sents = []
            else:
                sents.append(line)
        if sents:
            predict_and_print(sents)
