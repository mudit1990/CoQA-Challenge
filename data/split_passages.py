import json
from collections import defaultdict

with open('./train/coqa-train-v1.0.json') as jsonData:
    contents = json.load(jsonData)
    passages = contents['data']
    print('Number of passages in train = {}'.format(len(passages)))

    D = defaultdict(int)
    for passage in passages:
        D[passage['source']] += 1
    print('Number of passages in each domain: ')
    for key, value in D.items():
        print('{}\t{}'.format(key, value))
    
    reduced_train = {}
    reduced_train['version'] = '1.5'
    reduced_train['data'] = []

    testing_set   = {}
    testing_set['version'] = '1.5'
    testing_set['data'] = []

    D = defaultdict(int)
    for passage in passages:
        domain = passage['source']
        if D[domain] < 150:
            testing_set['data'].append(passage)
            D[domain] += 1
        else:
            reduced_train['data'].append(passage)

    training_file = './train/train.json'
    testing_file  =   './test/test.json'

    with open(training_file, 'w') as f:
        json.dump(reduced_train, f, indent=2, separators=(',', ':'))

    with open(testing_file, 'w') as f:
        json.dump(testing_set, f, indent=2, separators=(',', ':'))

# Confirm split
with open('./train/train.json') as f:
    contents = json.load(f)
    passages = contents['data']
    print('Number of passages in reduced train = {}'.format(len(passages)))
    D = defaultdict(int)
    for passage in passages:
        D[passage['source']] += 1
    print('Number of passages in each domain: ')
    for key, value in D.items():
        print('{}\t{}'.format(key, value))

with open('./test/test.json') as f:
    contents = json.load(f)
    passages = contents['data']
    print('Number of passages in reduced train = {}'.format(len(passages)))
    D = defaultdict(int)
    for passage in passages:
        D[passage['source']] += 1
    print('Number of passages in each domain: ')
    for key, value in D.items():
        print('{}\t{}'.format(key, value))
