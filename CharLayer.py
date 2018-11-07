import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Input : (N, sentence_length, word_length, char_vocab_size)
Out: (N, sentence_length, char_embd_size)
'''

class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.embd_size = args.char_embd_size

        #Create an embedding layer for the characters
        self.embedding = nn.Embedding(args.vocabc_size, args.char_embd_size)

        #Create a conv layer with dropout
        self.conv = nn.Conv2d(in_channels=1, out_channels=args.out_channels, (f[0], f[1])) for f in args.filters])
        self.dropout = nn.Dropout(0.4)
        
    '''
    Input: x: (N, sent_len, word_len)
    '''
    def forward(self, x):
        input_shape = x.size()
        batch_size = x.size(0)
        sent_len = x.size(1)
        word_len = x.size(2)
        #(N*sent_len, word_len)
        x = x.view(-1, word_len)
        #(N*sent_len, word_len, char_embd_size)
        x = self.embedding(x)
        #(N, sent_len, word_len, c_embd_size)
        x = x.view(*input_shape, -1)
        #(N, sent_len, c_embd_size)
        x = x.sum(2)

        #(N, in_channels, sent_len, c_embd_size)
        x = x.unsqueeze(1)
        
        x=F.relu(self.conv(x))
