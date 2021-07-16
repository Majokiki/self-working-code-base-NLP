# —*- coding: utf-8 -*-
class Config(object):
    def __init__(self, 
                word_embedding_dimension=100, 
                word_num=11000,
                cuda=False,
                batch_size=1,
                epoch=2, 
                learning_rate=0.001,
                sentence_max_size=25, 
                block_size=20,
                overlap=2,
                label_num=2, 
                kernel_num=100, 
                kernel_sizes=[3,4,5]):
        self.word_embedding_dimension = word_embedding_dimension     # 词向量的维度
        self.word_num = word_num
        self.epoch = epoch                                           # 遍历样本次数
        self.sentence_max_size = sentence_max_size                   # 句子长度
        self.label_num = label_num                                   # 分类标签个数
        self.lr = learning_rate
        self.batch_size = batch_size
        self.cuda = cuda
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.block_size = block_size
        self.overlap = overlap
        self.dropout = 0.5
        self.static = False
        self.lstm_hidden_dim = 300
        self.lstm_num_layers = 2
