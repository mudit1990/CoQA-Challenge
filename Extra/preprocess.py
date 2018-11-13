import numpy as np
import re
import torchtext.vocab as vocab
import torch
from collections import *
import simplejson as json

questions = []
answers = []

def myread(name):
    with open(name,'r') as trainfile:
        json_str = trainfile.read()
        json_data = json.loads(json_str)
        data = json_data['data']
        for i in data:
            print(i)
            inp = i['story']
            for j, k in zip(i['questions'],i['answers']):
                inp =inp+j['input_text']
                questions.append(inp)
                answers.append(k['span_text'])
        #print(questions[-1],answers[-1])

vocab_set = set()
char_set = set()
tag_set = set()

myread('coqa-train-v1.0.json')
