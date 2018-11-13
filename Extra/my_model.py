import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules.similarity_functions.linear import LinearSimilarity
HIDDEN_SIZE = 100
CEMBED_SIZE = 50

class BiDAF(nn.Module):
    def __init__(self, args, pretrained):
        super(BiDAF, self).__init__()
        self.args = args

        #1.CharCNN
        self.lookup_c = nn.Embedding(nchars, CEMBED_SIZE, padding_idx=CUNK)
        nn.init.uniform_(self.lookup_c, -0.001, 0.001)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_SIZE, kernel_size=(CEMBED_SIZE, 3))
        self.dropout = nn.Dropout(0.2)

        #Word Embeddings
        #Somehow initialize with elmo
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        
        #highway network
        for i in range(2):
            setattr(self, f'highway_linear{i}',nn.Sequential(Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2),nn.ReLU()))
            setattr(self, f'highway_gate{i}',nn.Sequential(Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2),nn.Sigmoid()))

        #Contextual Embedding Layer
        self.lstm = nn.LSTM(2 * HIDDEN_SIZE, HIDDEN_SIZE, dropout = 0.2, batch_first = True, bidirectional = True)

        #Attention Flow Layer
        self.c = nn.Linear(2 * HIDDEN_SIZE, 1, bias=False)
        self.Wq = nn.Linear(2 * HIDDEN_SIZE, 1, bias=False)
        self.Wcq = nn.Linear(2 * HIDDEN_SIZE, 1, bias=False)

        #modeling
        self.lstm2 = nn.LSTM(8 * HIDDEN_SIZE, HIDDEN_SIZE, dropout = 0.2, batch_first = True, bidirectional = True)

        #span end encoder
        self.lstm_end_encoder = nn.LSTM(14 * HIDDEN_SIZE, HIDDEN_SIZE, dropout = 0.2, batch_first = True, bidirectional = True)

        #Start and end span predictors
        self.linear_end = nn.Linear(10 * HIDDEN_SIZE, 1)
        self.linear_start = nn.Linear(10 * HIDDEN_SIZE, 1)
        self.dropout = nn.Dropout(0.2)



